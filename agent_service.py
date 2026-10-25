import logging
import requests
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(
    title="Agent Layer API - Генерация структуры документов (Вариант 6)",
    version="1.0.0",
    description="Сервис агентного уровня для генерации структуры документов через локальную LLM (Ollama) и передачи сегментов на транспортный уровень."
)

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3"  # Локальная модель 1-7 млрд параметров
TRANSPORT_TRANSFER_URL = "http://localhost:8080/api/v1/transfer"
SEGMENT_SIZE_BYTES = 512  # Фиксированный размер сегмента (X байт)


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


def generate_and_dispatch_segments(task: AppMessage) -> int:
    """Синхронно генерирует структуру в LLM и передает все сегменты в транспорт."""
    logging.info(f"Запуск генерации структуры по теме: '{task.document_topic}' (timestamp: {task.timestamp})")

    system_prompt = (
        "Ты — технический архитектор. Создай подробную структуру документа (оглавление, разделы, подразделы) "
        "на заданную тему. Форматируй ответ строго как нумерованный список разделов с кратким пояснением каждого пункта."
    )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"{system_prompt}\n\nТема документа: {task.document_topic}",
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        response.raise_for_status()
        full_text = response.json().get("response", "")
    except requests.RequestException as e:
        logging.error(f"Ошибка обращения к локальной Ollama: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Сбой работы локальной LLM: {e}"
        )

    raw_segments = split_text_by_bytes(full_text, max_bytes=SEGMENT_SIZE_BYTES)
    total_segments = len(raw_segments)
    logging.info(f"Генерация завершена. Сформировано сегментов: {total_segments} (по {SEGMENT_SIZE_BYTES} байт)")

    for idx, chunk in enumerate(raw_segments, start=1):
        segment_payload = AgentSegment(
            timestamp=task.timestamp,
            total_segments=total_segments,
            segment_number=idx,
            document_structure_segment=chunk
        )

        try:
            transfer_res = requests.post(
                TRANSPORT_TRANSFER_URL,
                json=segment_payload.model_dump(),
                timeout=5
            )
            logging.info(
                f"Сегмент {idx}/{total_segments} передан на транспортный уровень. "
                f"Статус ответа: {transfer_res.status_code}"
            )
        except requests.RequestException as e:
            logging.error(f"Сбой отправки сегмента {idx} на транспортный уровень: {e}")

    return total_segments


# --- Эндпоинты сервиса ---

@app.post(
    "/api/v1/generate",
    status_code=status.HTTP_200_OK,
    response_model=OperationResponse,
    summary="Генерация структуры документа и отправка сегментов в транспорт",
    description="Принимает тему документа, синхронно генерирует структуру через LLM, передает сегменты на транспорт и возвращает 200 OK по окончании."
)
def generate_document_structure(task: AppMessage):
    # Синхронно выполняем инференс и отправку
    total = generate_and_dispatch_segments(task)
    
    # Ответ 200 OK возвращается ТОЛЬКО после того, как все сегменты отправлены
    return OperationResponse(
        success=True,
        message=f"Генерация завершена. Успешно сформировано и передано {total} сегментов."
    )


@app.get(
    "/api/v1/health",
    summary="Проверка статуса сервиса и доступности локальной LLM"
)
def health_check():
    try:
        res = requests.get("http://localhost:11434/api/tags", timeout=3)
        ollama_status = "connected" if res.status_code == 200 else "unavailable"
    except Exception:
        ollama_status = "offline"

    return {
        "service": "Agent Layer (Variant 6)",
        "status": "ok",
        "ollama_status": ollama_status,
        "model": OLLAMA_MODEL,
        "segment_size_bytes": SEGMENT_SIZE_BYTES
    }