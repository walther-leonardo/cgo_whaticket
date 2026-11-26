
from __future__ import annotations

import logging
import os
import re
import time
import unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from uuid import UUID
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================
# CONFIGURACIÓN
# ============================================================

ROOT = Path(__file__).resolve().parent
CEL_FILE = ROOT / "cel.txt"
CSV_FILE = ROOT / "crudo.csv"
OUTPUT_DIR = ROOT / "resultados"

API_URL = "https://api.whaticket.com/api/v1"
LIMA = ZoneInfo("America/Lima")

FECHA_INICIO = datetime(2026, 11, 7, tzinfo=LIMA)
FECHA_FIN = datetime(2026, 11, 12, tzinfo=LIMA)

MARCADOR_CAMPANA = "sorteamos 10 mochilas"
MAX_PAGINAS = 250

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
log = logging.getLogger("cgo_whaticket")


# ============================================================
# NORMALIZACIÓN
# ============================================================

def normalizar(texto: object) -> str:
    valor = unicodedata.normalize("NFKD", str(texto or ""))
    valor = "".join(
        c for c in valor
        if not unicodedata.combining(c)
    )
    return " ".join(valor.casefold().split())


def normalizar_telefono(valor: object) -> str:
    numero = re.sub(r"\D", "", str(valor or ""))

    # Números móviles peruanos con y sin código de país.
    if len(numero) == 9 and numero.startswith("9"):
        numero = "51" + numero

    return numero


def fecha_local(valor: str) -> datetime:
    fecha = datetime.fromisoformat(valor.replace("Z", "+00:00"))
    if fecha.tzinfo is None:
        raise ValueError(f"Fecha sin zona horaria: {valor}")
    return fecha.astimezone(LIMA)


def formato_fecha(fecha: datetime) -> str:
    return fecha.strftime("%Y-%m-%d %H:%M:%S")


def dentro_periodo(mensaje: dict) -> bool:
    fecha = fecha_local(mensaje["createdAt"])
    return FECHA_INICIO <= fecha < FECHA_FIN


def leer_telefonos() -> list[str]:
    if not CEL_FILE.exists():
        raise FileNotFoundError(f"No existe {CEL_FILE}")

    telefonos = []
    vistos = set()

    for numero_linea, linea in enumerate(
        CEL_FILE.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        valor = linea.strip()

        if not valor or valor.startswith("#"):
            continue

        telefono = normalizar_telefono(valor)

        if not re.fullmatch(r"519\d{8}", telefono):
            raise ValueError(
                f"Teléfono inválido en cel.txt, "
                f"línea {numero_linea}: {valor}"
            )

        if telefono not in vistos:
            vistos.add(telefono)
            telefonos.append(telefono)

    if not telefonos:
        raise ValueError("cel.txt no contiene teléfonos válidos.")

    return telefonos


# ============================================================
# CRUCE CON REPORTE WHATICKET
# ============================================================

def leer_reporte() -> tuple[pd.DataFrame, int]:
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"No existe {CSV_FILE}")

    df = pd.read_csv(
        CSV_FILE,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
    )

    df.columns = df.columns.str.strip()
    filas_originales = len(df)

    requeridas = {"contactNumber", "chatId"}

    if not requeridas.issubset(df.columns):
        faltantes = requeridas - set(df.columns)
        raise ValueError(
            f"Faltan columnas en crudo.csv: {sorted(faltantes)}"
        )

    if "contactName" not in df.columns:
        df["contactName"] = ""

    df["telefono"] = df["contactNumber"].map(
        normalizar_telefono
    )

    df["chatId"] = df["chatId"].str.strip()

    # Se eliminan únicamente filas sin chatId y duplicados
    # exactos de teléfono + chatId.
    df = df[df["chatId"] != ""].copy()

    df = df.drop_duplicates(
        subset=["telefono", "chatId"],
        keep="last",
    )

    return df, filas_originales


def construir_indices(
    df: pd.DataFrame,
    telefonos: list[str],
) -> tuple[dict, dict]:
    universo = set(telefonos)

    chats_por_telefono = defaultdict(list)
    telefonos_por_chat = defaultdict(set)

    for registro in df.itertuples(index=False):
        telefono = registro.telefono
        chat_id = registro.chatId

        if telefono not in universo:
            continue

        chats_por_telefono[telefono].append({
            "chatId": chat_id,
            "contactName": registro.contactName,
        })

        telefonos_por_chat[chat_id].add(telefono)

    return chats_por_telefono, telefonos_por_chat


# ============================================================
# API WHATICKET
# ============================================================

def crear_sesion(token: str) -> requests.Session:
    sesion = requests.Session()

    sesion.headers.update({
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    })

    retry = Retry(
        total=4,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )

    sesion.mount(
        "https://",
        HTTPAdapter(max_retries=retry),
    )

    return sesion


def obtener_mensajes(
    sesion: requests.Session,
    chat_id: str,
) -> list[dict]:
    url = f"{API_URL}/tickets/{chat_id}/messages"

    parametros = {}
    encontrados = {}
    cursores = set()

    for _ in range(MAX_PAGINAS):
        respuesta = sesion.get(
            url,
            params=parametros,
            timeout=30,
        )
        respuesta.raise_for_status()

        datos = respuesta.json()
        pagina = datos.get("messages", [])

        if not isinstance(pagina, list):
            raise ValueError("Respuesta API inválida: messages")

        for mensaje in pagina:
            mensaje_id = mensaje.get("id")

            if not mensaje_id:
                raise ValueError("Mensaje sin identificador")

            encontrados[mensaje_id] = mensaje

        if not datos.get("hasMoreBefore", False):
            return sorted(
                encontrados.values(),
                key=lambda m: (
                    fecha_local(m["createdAt"]),
                    m["id"],
                ),
            )

        if not pagina:
            raise ValueError(
                "Paginación incompleta: página vacía"
            )

        primero = pagina[0]
        cursor = (
            primero["createdAt"],
            primero["id"],
        )

        if cursor in cursores:
            raise ValueError("Cursor de paginación repetido")

        cursores.add(cursor)

        parametros = {
            "cursor": cursor[0],
            "cursorId": cursor[1],
            "direction": "before",
        }

        time.sleep(0.12)

    raise RuntimeError(
        f"Límite de paginación alcanzado: {chat_id}"
    )


# ============================================================
# IDENTIFICACIÓN DE CAMPAÑA Y RESPUESTAS
# ============================================================

def es_invitacion(mensaje: dict) -> bool:
    return (
        mensaje.get("fromMe") is True
        and not mensaje.get("isNote")
        and not mensaje.get("isDeleted")
        and dentro_periodo(mensaje)
        and normalizar(MARCADOR_CAMPANA)
        in normalizar(mensaje.get("body", ""))
    )


def es_flow(mensaje: dict) -> bool:
    return (
        mensaje.get("fromMe") is False
        and normalizar(mensaje.get("mediaType")) == "flowresponse"
        and not mensaje.get("isDeleted")
        and not mensaje.get("isNote")
    )


def clasificar(cuerpo: str) -> tuple[str, str, str]:
    limpio = re.sub(
        r"^\s*opci[oó]n\s+[uú]nica\s*:\s*",
        "",
        cuerpo or "",
        flags=re.IGNORECASE,
    ).strip()

    texto = normalizar(limpio)

    if re.fullmatch(
        r"si\s+aun\s+soy\s+propietario",
        texto,
    ):
        return "Propietario actual", "", limpio

    match = re.fullmatch(
        r"no\s+soy\s+propietario\s+desde\s+(20\d{2})",
        texto,
    )

    if match and 2021 <= int(match.group(1)) <= 2026:
        return "Expropietario", match.group(1), limpio

    return "No reconocida", "", limpio


def procesar_chat(
    chat_id: str,
    mensajes: list[dict],
) -> tuple[dict, list[dict], list[dict]]:

    enlace = f"https://app.whaticket.com/tickets/{chat_id}"

    mensajes_periodo = [
        m for m in mensajes
        if dentro_periodo(m)
    ]

    invitaciones = [
        m for m in mensajes_periodo
        if es_invitacion(m)
    ]

    info = {
        "chatId": chat_id,
        "chatLink": enlace,
        "Estado_chat": "Sin invitacion",
        "Invitaciones": len(invitaciones),
        "Respuestas_validas": 0,
        "Flows_no_reconocidos": 0,
        "Mensajes_periodo": len(mensajes_periodo),
        "Error": "",
    }

    if not invitaciones:
        return info, [], []

    # Identificamos respuestas posteriores a la primera
    # invitación. Si hay otra invitación, seguimos conservando
    # todos los eventos para revisión.
    primera_fecha = fecha_local(
        invitaciones[0]["createdAt"]
    )

    mensajes_campana = [
        m for m in mensajes_periodo
        if fecha_local(m["createdAt"]) >= primera_fecha
    ]

    respuestas = []
    auditoria = []

    for mensaje in mensajes_campana:
        fecha = fecha_local(mensaje["createdAt"])

        auditoria.append({
            "chatId": chat_id,
            "Fecha_Lima": formato_fecha(fecha),
            "messageId": mensaje.get("id", ""),
            "Direccion": (
                "PanaAutos" if mensaje.get("fromMe")
                else "Cliente"
            ),
            "mediaType": mensaje.get("mediaType") or "",
            "body": str(mensaje.get("body") or "")[:32767],
        })

        if not es_flow(mensaje):
            continue

        categoria, anio, respuesta = clasificar(
            mensaje.get("body", "")
        )

        if categoria == "No reconocida":
            info["Flows_no_reconocidos"] += 1
        else:
            info["Respuestas_validas"] += 1

        respuestas.append({
            "chatId": chat_id,
            "chatLink": enlace,
            "messageId": mensaje["id"],
            "Fecha_respuesta": formato_fecha(fecha),
            "Respuesta": respuesta,
            "Clasificacion": categoria,
            "Anio_venta": anio,
        })

    if info["Respuestas_validas"]:
        info["Estado_chat"] = "Respondida"
    elif info["Flows_no_reconocidos"]:
        info["Estado_chat"] = "Flow por revisar"
    else:
        info["Estado_chat"] = "Sin respuesta Flow"

    return info, respuestas, auditoria


# ============================================================
# EXCEL
# ============================================================

def guardar_excel(
    contactos: list[dict],
    chats: list[dict],
    respuestas: list[dict],
    mensajes: list[dict],
) -> Path:

    OUTPUT_DIR.mkdir(exist_ok=True)

    nombre = (
        "encuesta_2do_cliente_"
        + datetime.now(LIMA).strftime("%Y%m%d_%H%M%S")
        + ".xlsx"
    )

    destino = OUTPUT_DIR / nombre

    hojas = {
        "Contactos": pd.DataFrame(contactos),
        "Chats": pd.DataFrame(chats),
        "Respuestas": pd.DataFrame(respuestas),
        "Mensajes": pd.DataFrame(mensajes),
    }

    with pd.ExcelWriter(destino, engine="openpyxl") as writer:
        for nombre_hoja, df in hojas.items():
            df.to_excel(
                writer,
                sheet_name=nombre_hoja,
                index=False,
            )

            ws = writer.sheets[nombre_hoja]
            ws.freeze_panes = "A2"

            if ws.max_row > 1:
                ws.auto_filter.ref = ws.dimensions

            for columna in ws.columns:
                letra = columna[0].column_letter
                muestra = list(columna)[:101]

                ancho = max(
                    len(str(celda.value or ""))
                    for celda in muestra
                ) + 2

                ws.column_dimensions[letra].width = min(
                    max(ancho, 14),
                    55,
                )

    return destino


# ============================================================
# EJECUCIÓN
# ============================================================


def leer_ids() -> list[str]:
    """Lee chatIds desde ids.txt, validando y quitando duplicados."""
    archivo = ROOT / "ids.txt"

    if not archivo.exists():
        raise FileNotFoundError(f"No existe {archivo}")

    ids = []
    vistos = set()

    for numero, linea in enumerate(
        archivo.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        valor = linea.strip()

        if not valor or valor.startswith("#"):
            continue

        try:
            chat_id = str(UUID(valor))
        except ValueError as exc:
            raise ValueError(
                f"ID inválido en línea {numero}: {valor}"
            ) from exc

        if chat_id not in vistos:
            vistos.add(chat_id)
            ids.append(chat_id)

    if not ids:
        raise ValueError("ids.txt no contiene IDs válidos.")

    return ids


def main() -> None:
    load_dotenv(ROOT / ".env")

    token = os.getenv("WHATICKET_TOKEN", "").strip()

    if not token:
        raise RuntimeError(
            "No se encontró WHATICKET_TOKEN en .env"
        )

    ids = leer_ids()

    log.info("=" * 60)
    log.info("CGO WHATICKET — ENCUESTA 2DO CLIENTE")
    log.info("=" * 60)
    log.info("Chats únicos por consultar: %s", len(ids))

    contactos = []
    todos_chats = []
    todas_respuestas = []
    todos_mensajes = []

    with crear_sesion(token) as sesion:
        for indice, chat_id in enumerate(ids, start=1):
            log.info(
                "[%s/%s] %s",
                indice,
                len(ids),
                chat_id,
            )

            try:
                mensajes = obtener_mensajes(sesion, chat_id)

                info, respuestas, auditoria = procesar_chat(
                    chat_id,
                    mensajes,
                )

            except requests.HTTPError as exc:
                codigo = exc.response.status_code

                if codigo == 401:
                    raise RuntimeError(
                        "Token inválido o vencido."
                    ) from exc

                info = {
                    "chatId": chat_id,
                    "Estado_chat": "Error API",
                    "Error": f"HTTP {codigo}",
                }
                respuestas, auditoria = [], []

            except (
                requests.RequestException,
                ValueError,
                RuntimeError,
            ) as exc:
                info = {
                    "chatId": chat_id,
                    "Estado_chat": "Error API",
                    "Error": str(exc),
                }
                respuestas, auditoria = [], []

            todos_chats.append(info)
            todas_respuestas.extend(respuestas)
            todos_mensajes.extend(auditoria)

            validas = [
                respuesta
                for respuesta in respuestas
                if respuesta["Clasificacion"] != "No reconocida"
            ]

            validas.sort(
                key=lambda r: (
                    r["Fecha_respuesta"],
                    r["messageId"],
                )
            )

            ultima = validas[-1] if validas else {}

            error = info.get("Error", "")

            contactos.append({
                "chatId": chat_id,
                "chatLink": (
                    f"https://app.whaticket.com/tickets/{chat_id}"
                ),
                "Estado": info["Estado_chat"],
                "Clasificacion": ultima.get(
                    "Clasificacion", ""
                ),
                "Anio_venta": ultima.get(
                    "Anio_venta", ""
                ),
                "Respuesta": ultima.get(
                    "Respuesta", ""
                ),
                "Fecha_respuesta": ultima.get(
                    "Fecha_respuesta", ""
                ),
                "Invitaciones": info.get(
                    "Invitaciones", 0
                ),
                "Respuestas_validas": len(validas),
                "Flows_no_reconocidos": info.get(
                    "Flows_no_reconocidos", 0
                ),
                "Revisar": (
                    "Sí"
                    if len(respuestas) > 1
                    or info.get("Flows_no_reconocidos", 0) > 0
                    or info["Estado_chat"] == "Error API"
                    else "No"
                ),
                "Error": error,
            })

            log.info(
                "  %s | respuestas válidas: %s",
                info["Estado_chat"],
                len(validas),
            )

    destino = guardar_excel(
        contactos,
        todos_chats,
        todas_respuestas,
        todos_mensajes,
    )

    df = pd.DataFrame(contactos)

    propietarios = (
        df["Clasificacion"] == "Propietario actual"
    ).sum()

    expropietarios = (
        df["Clasificacion"] == "Expropietario"
    ).sum()

    respondidos = (
        df["Estado"] == "Respondida"
    ).sum()

    log.info("=" * 60)
    log.info("RESULTADOS — ENCUESTA 2DO CLIENTE")
    log.info("=" * 60)
    log.info("Chats consultados: %s", len(ids))
    log.info("Encuestas respondidas: %s", respondidos)
    log.info("Propietarios actuales: %s", propietarios)
    log.info("Expropietarios: %s", expropietarios)

    for estado, cantidad in df["Estado"].value_counts().items():
        log.info("%s: %s", estado, cantidad)

    log.info(
        "Registros marcados para revisión: %s",
        (df["Revisar"] == "Sí").sum(),
    )

    log.info("Excel generado: %s", destino)


if __name__ == "__main__":
    main()
