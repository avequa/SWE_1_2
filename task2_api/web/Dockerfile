FROM golang:1.27-alpine AS build
WORKDIR /src
COPY go.mod main.go index.html ./
RUN CGO_ENABLED=0 go build -o /web .


FROM alpine:3.24
COPY --from=build /web /web
EXPOSE 8080
CMD ["/web"]
