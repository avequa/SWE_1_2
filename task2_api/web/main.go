// Пользователь вводит отзывы, сервис отправляет их в ML API
// (FastAPI, POST /predict/batch) и показывает тональность каждого отзыва, а также сводку за всю сессию
// Сервер ничего не запоминает между запросами

// Запуск:   ML_API_URL=http://localhost:8000 go run .
// Страница: http://localhost:8080

package main

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"fmt"
	"html/template"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

const maxReviews = 100 // столько же принимает ML API за один запрос

//go:embed index.html
var pageHTML string

// html/template сам экранирует текст отзывов, поэтому HTML из формы не выполнится
var page = template.Must(template.New("page").Parse(pageHTML))

var labelsRU = map[string]string{
	"positive": "позитивный",
	"neutral":  "нейтральный",
	"negative": "негативный",
}

var (
	apiURL     = getenv("ML_API_URL", "http://localhost:8000")
	grafanaURL = getenv("GRAFANA_URL", "http://localhost:3000/d/sentiment-api")
	docsURL    = getenv("API_DOCS_URL", "http://localhost:8000/docs")
	client     = &http.Client{Timeout: 30 * time.Second}
)

// Ответ ML API на POST /predict/batch
type Prediction struct {
	Label  string             `json:"label"`
	Scores map[string]float64 `json:"scores"`
}

type BatchResponse struct {
	Results []Prediction   `json:"results"`
	Summary map[string]int `json:"summary"`
}

// Counts - сколько кол-во отзывов каждого класса
type Counts struct {
	Positive, Neutral, Negative int
}

func (c Counts) Total() int { return c.Positive + c.Neutral + c.Negative }

// Percent - доля класса в процентах, для подписей под шкалой
func (c Counts) Percent(n int) int {
	if c.Total() == 0 {
		return 0
	}
	return (n*100 + c.Total()/2) / c.Total()
}

// для HTML-шаблона
type Row struct {
	Text       string
	Label      string
	LabelRU    string
	Confidence int // уверенность модели в процентах
}

type PageData struct {
	Input      string
	Rows       []Row
	Session    Counts // сводка за всю сессию
	Error      string
	GrafanaURL string
	DocsURL    string
}

func main() {
	http.HandleFunc("GET /{$}", showForm)
	http.HandleFunc("POST /{$}", analyze)

	log.Printf("веб-интерфейс: http://localhost:8080, ML API: %s", apiURL)
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func showForm(w http.ResponseWriter, r *http.Request) {
	render(w, PageData{})
}

func analyze(w http.ResponseWriter, r *http.Request) {
	input := r.FormValue("reviews")
	data := PageData{Input: input, Session: sessionFromForm(r)}

	var texts []string
	for _, line := range strings.Split(input, "\n") {
		if line = strings.TrimSpace(line); line != "" {
			texts = append(texts, line)
		}
	}
	if len(texts) == 0 {
		data.Error = "Введите хотя бы один отзыв: каждый с новой строки"
		render(w, data)
		return
	}
	if len(texts) > maxReviews {
		data.Error = fmt.Sprintf("За раз можно проверить до %d отзывов, а введено %d", maxReviews, len(texts))
		render(w, data)
		return
	}

	start := time.Now()
	result, err := predictBatch(texts)
	if err != nil {
		log.Printf("ошибка ML API: %v", err)
		data.Error = "Сервис распознавания сейчас недоступен. Отзывы сохранены в поле, попробуйте ещё раз через минуту."
		render(w, data)
		return
	}
	log.Printf("отзывов: %d, итог: %v, время: %s", len(texts), result.Summary, time.Since(start).Round(time.Millisecond))

	for i, p := range result.Results {
		data.Rows = append(data.Rows, Row{
			Text:       texts[i],
			Label:      p.Label,
			LabelRU:    labelsRU[p.Label],
			Confidence: int(p.Scores[p.Label]*100 + 0.5),
		})
	}

	// чтобы при следующей проверке те же отзывы не посчитались второй раз
	data.Session.Positive += result.Summary["positive"]
	data.Session.Neutral += result.Summary["neutral"]
	data.Session.Negative += result.Summary["negative"]
	data.Input = ""
	render(w, data)
}

// sessionFromForm читает сводку сессии из скрытых полей формы
func sessionFromForm(r *http.Request) Counts {
	return Counts{
		Positive: formInt(r, "session_positive"),
		Neutral:  formInt(r, "session_neutral"),
		Negative: formInt(r, "session_negative"),
	}
}

func formInt(r *http.Request, name string) int {
	n, err := strconv.Atoi(r.FormValue(name))
	if err != nil || n < 0 {
		return 0
	}
	return n
}

// predictBatch отправляет отзывы в ML API и разбирает ответ
func predictBatch(texts []string) (*BatchResponse, error) {
	body, err := json.Marshal(map[string][]string{"texts": texts})
	if err != nil {
		return nil, err
	}
	resp, err := client.Post(apiURL+"/predict/batch", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("код ответа %d", resp.StatusCode)
	}
	var result BatchResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	return &result, nil
}

func render(w http.ResponseWriter, data PageData) {
	data.GrafanaURL = grafanaURL
	data.DocsURL = docsURL
	if err := page.Execute(w, data); err != nil {
		log.Printf("ошибка шаблона: %v", err)
	}
}

func getenv(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}
