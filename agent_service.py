import logging
import os
import re
import time
from logging.handlers import RotatingFileHandler

import requests
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# --- Настройка логирования (консоль + файл с ротацией) ---

LOG_FILE = "agent_service.log"
LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"

logger = logging.getLogger("agent_layer")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    formatter = logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,  # 5 МБ на файл
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

app = FastAPI(
    title="Agent Layer API - Генерация структуры документов (Вариант 6)",
    version="1.0.0",
    description="Сервис агентного уровня для генерации структуры документов через локальную LLM (Ollama) и передачи сегментов на транспортный уровень.",
)

# Используем прямой IP 127.0.0.1 для исключения IPv6-задержек на Windows
OLLAMA_CHAT_URL = os.getenv("OLLAMA_CHAT_URL", "http://127.0.0.1:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
TRANSPORT_TRANSFER_URL = os.getenv(
    "TRANSPORT_TRANSFER_URL", "http://10.170.95.181:8080/api/v1/transfer"
)
SEGMENT_SIZE_BYTES = 512


# --- Pydantic Схемы данных ---

class AppMessage(BaseModel):
    sender: str = Field(..., example="Ivan_Student", description="Имя пользователя")
    timestamp: int = Field(..., example=1715600000, description="Идентификатор запроса и временная метка")
    document_topic: str = Field(..., example="Архитектура распределенных высоконагруженных систем", description="Тема документа")


class OperationResponse(BaseModel):
    success: bool = Field(True, example=True)
    message: str = Field(..., example="Генерация и передача всех сегментов успешно завершена")


class AgentSegment(BaseModel):
    timestamp: int
    total_segments: int
    segment_number: int
    document_structure_segment: str


# --- Вспомогательные функции ---

def clean_generated_structure(text: str) -> str:
    """Убирает случайную Markdown-разметку, повторы вида '6. Раздел 6:' и иероглифы."""
    cleaned_lines = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue

        # Убираем Markdown (#, **, __)
        line = re.sub(r"^\s*#+\s*", "", line)
        line = line.replace("**", "").replace("__", "")

        # Убираем случайные китайские иероглифы (артефакт qwen2.5)
        line = re.sub(r"[\u4e00-\u9fff\u3000-\u303f]+", "", line)

        # Убираем дубли вида "6. Раздел 6: Заголовок" или "7. Раздел 7: Раздел 7: Заголовок"
        line = re.sub(
            r"^(\s*\d+\.)\s*(?:(?:Раздел|Глава|Часть)\s*\d*\s*[:.\-—]?\s*)+",
            r"\1 ",
            line,
            flags=re.IGNORECASE,
        )

        # Нормализуем двойные пробелы внутри строки (сохраняя отступ подразделов)
        indent_match = re.match(r"^(\s*)", line)
        indent = indent_match.group(1) if indent_match else ""
        body = re.sub(r"\s{2,}", " ", line.strip())
        cleaned_lines.append(f"{indent}{body}")

    return "\n".join(cleaned_lines)


def split_text_by_bytes(text: str, max_bytes: int = 512) -> list[str]:
    """Разбивает строку на части так, чтобы каждая часть в UTF-8 не превышала max_bytes."""
    encoded = text.encode("utf-8")
    chunks = []
    start = 0
    while start < len(encoded):
        end = start + max_bytes
        chunk_bytes = encoded[start:end]
        chunk_str = chunk_bytes.decode("utf-8", errors="ignore")
        actual_bytes_len = len(chunk_str.encode("utf-8"))
        if actual_bytes_len == 0:
            break
        chunks.append(chunk_str)
        start += actual_bytes_len
    return chunks


def generate_and_dispatch_segments(task: AppMessage) -> tuple[int, int]:
    """Генерирует двухуровневую структуру через Chat API и передает сегменты через HTTP-сессию."""
    total_start = time.perf_counter()
    logger.info(
        "[ts=%s] Входящий запрос | sender='%s' | topic='%s'",
        task.timestamp,
        task.sender,
        task.document_topic,
    )

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты составляешь оглавление документа на русском языке.\n"
                    "Выведи строго 7 пронумерованных глав (от 1 до 7), в каждой ровно 3 подпункта.\n\n"
                    "Правила оформления:\n"
                    "- Пиши ТОЛЬКО на русском языке кириллицей (без английских слов и иероглифов).\n"
                    "- После цифры сразу пиши короткий смысловой заголовок (2–5 слов). "
                    "ЗАПРЕЩЕНО писать слово «Раздел», «Глава» или повторять номер.\n"
                    "- Не повторяй формулировку темы в каждом заголовке.\n"
                    "- В каждом подпункте — краткое название и пояснение в 1 короткое предложение.\n"
                    "- Без Markdown (#, **), без вступлений и заключений.\n\n"
                    "Пример правильного формата (для темы «Устройство велосипеда»):\n"
                    "1. Общие сведения и классификация\n"
                    "   1.1. История развития конструкции. Краткий обзор эволюции велосипедов.\n"
                    "   1.2. Основные типы моделей. Сравнение шоссейных, горных и городских вариантов.\n"
                    "   1.3. Области применения. Использование в спорте, туризме и повседневных поездках.\n"
                    "2. Рама и ходовая часть\n"
                    "   2.1. Геометрия и материалы рамы. Влияние сплава на вес и жёсткость.\n"
                    "   2.2. Колёса и покрышки. Выбор диаметра обода и рисунка протектора.\n"
                    "   2.3. Амортизационная система. Принцип работы вилок и задних амортизаторов."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Составь оглавление по теме: «{task.document_topic}».\n"
                    "Строго 7 пунктов (с 1 по 7), в каждом по 3 подпункта (до 7.3 включительно). "
                    "Только русский язык, без слова «Раздел» в заголовках."
                ),
            },
        ],
        "stream": False,
        "options": {
            "temperature": 0.3,
            "repeat_penalty": 1.15,
            "num_predict": 2500,
            "num_ctx": 8192,
        },
    }

    llm_start = time.perf_counter()
    logger.info("[ts=%s] Отправка запроса в Ollama (%s)...", task.timestamp, OLLAMA_MODEL)

    try:
        response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=120)
        response.raise_for_status()
        llm_data = response.json()
        raw_text = llm_data.get("message", {}).get("content", "")
    except requests.RequestException as e:
        logger.exception("[ts=%s] Ошибка обращения к Ollama: %s", task.timestamp, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Сбой работы локальной LLM: {e}",
        )

    llm_elapsed = time.perf_counter() - llm_start
    eval_count = llm_data.get("eval_count", "?")
    done_reason = llm_data.get("done_reason", "?")
    logger.info(
        "[ts=%s] Ответ Ollama получен за %.2f с | токенов: %s | done_reason: %s",
        task.timestamp,
        llm_elapsed,
        eval_count,
        done_reason,
    )

    full_text = clean_generated_structure(raw_text)
    if not full_text:
        logger.error("[ts=%s] LLM вернула пустой ответ", task.timestamp)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Локальная LLM вернула пустую структуру документа",
        )

    total_bytes = len(full_text.encode("utf-8"))
    logger.info(
        "[ts=%s] Сгенерированный текст (%d симв., %d байт):\n%s",
        task.timestamp,
        len(full_text),
        total_bytes,
        full_text,
    )

    raw_segments = split_text_by_bytes(full_text, max_bytes=SEGMENT_SIZE_BYTES)
    total_segments = len(raw_segments)
    delivered_segments = 0
    logger.info(
        "[ts=%s] Сегментация завершена: %d сегментов (макс. %d байт)",
        task.timestamp,
        total_segments,
        SEGMENT_SIZE_BYTES,
    )

    transfer_start = time.perf_counter()
    with requests.Session() as session:
        for idx, chunk in enumerate(raw_segments, start=1):
            chunk_bytes = len(chunk.encode("utf-8"))
            segment_payload = AgentSegment(
                timestamp=task.timestamp,
                total_segments=total_segments,
                segment_number=idx,
                document_structure_segment=chunk,
            )

            seg_start = time.perf_counter()
            try:
                transfer_res = session.post(
                    TRANSPORT_TRANSFER_URL,
                    json=segment_payload.model_dump(),
                    timeout=5,
                )
                seg_ms = (time.perf_counter() - seg_start) * 1000
                if transfer_res.status_code == status.HTTP_200_OK:
                    delivered_segments += 1
                    logger.info(
                        "[ts=%s] Сегмент %d/%d (%d байт) -> %s | HTTP 200 OK (%.1f мс)",
                        task.timestamp,
                        idx,
                        total_segments,
                        chunk_bytes,
                        TRANSPORT_TRANSFER_URL,
                        seg_ms,
                    )
                else:
                    logger.warning(
                        "[ts=%s] Сегмент %d/%d (%d байт) отклонен транспортом | HTTP %d (%.1f мс) | body=%s",
                        task.timestamp,
                        idx,
                        total_segments,
                        chunk_bytes,
                        transfer_res.status_code,
                        seg_ms,
                        transfer_res.text[:200],
                    )
            except requests.RequestException as e:
                logger.error(
                    "[ts=%s] Сбой отправки сегмента %d/%d на %s: %s",
                    task.timestamp,
                    idx,
                    total_segments,
                    TRANSPORT_TRANSFER_URL,
                    e,
                )

    transfer_elapsed = time.perf_counter() - transfer_start
    total_elapsed = time.perf_counter() - total_start
    logger.info(
        "[ts=%s] Итог: доставлено %d/%d сегментов | отправка: %.2f с | всего: %.2f с",
        task.timestamp,
        delivered_segments,
        total_segments,
        transfer_elapsed,
        total_elapsed,
    )

    return total_segments, delivered_segments


# --- Эндпоинты сервиса ---

@app.post(
    "/api/v1/generate",
    status_code=status.HTTP_200_OK,
    response_model=OperationResponse,
    summary="Генерация структуры документа и отправка сегментов в транспорт",
    description="Принимает тему документа, синхронно генерирует структуру глубины 2 через LLM, передает сегменты на транспорт и возвращает 200 OK по окончании.",
)
def generate_document_structure(task: AppMessage):
    total, delivered = generate_and_dispatch_segments(task)
    if delivered < total:
        logger.error(
            "[ts=%s] Возврат 502 Bad Gateway: передано %d из %d сегментов",
            task.timestamp,
            delivered,
            total,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Передано только {delivered} из {total} сегментов на транспортный уровень.",
        )

    return OperationResponse(
        success=True,
        message=f"Генерация завершена. Успешно сформировано и передано {delivered}/{total} сегментов.",
    )


@app.get(
    "/api/v1/health",
    summary="Проверка статуса сервиса и доступности локальной LLM",
)
def health_check():
    try:
        res = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
        ollama_status = "connected" if res.status_code == 200 else "unavailable"
    except Exception as e:
        logger.warning("Healthcheck: Ollama недоступна (%s)", e)
        ollama_status = "offline"

    logger.info("Healthcheck: status=ok, ollama_status=%s", ollama_status)
    return {
        "service": "Agent Layer (Variant 6)",
        "status": "ok",
        "ollama_status": ollama_status,
        "model": OLLAMA_MODEL,
        "segment_size_bytes": SEGMENT_SIZE_BYTES,
    }