COMPOSE_FILE = docker-compose.yaml
SERVICE_NAME = pagerank

.PHONY: all
all: up

.PHONY: up
up:
	@echo "Starting services with Docker Compose..."
	docker compose -f $(COMPOSE_FILE) up -d

.PHONY: down
down:
	@echo "Stopping services with Docker Compose..."
	docker compose -f $(COMPOSE_FILE) down

.PHONY: rebuild
rebuild:
	@echo "Rebuilding and starting services..."
	docker compose -f $(COMPOSE_FILE) up --build -d

.PHONY: shell
shell:
	@echo "Opening a shell in the $(SERVICE_NAME) container..."
	docker exec -it $(SERVICE_NAME) bash

.PHONY: logs
logs:
	@echo "Showing logs for $(SERVICE_NAME)..."
	docker logs $(SERVICE_NAME) -f

.PHONY: clean
clean:
	@echo "Cleaning up resources..."
	docker compose -f $(COMPOSE_FILE) down --volumes --remove-orphans

.PHONY: restart
restart: down up
