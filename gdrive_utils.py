from __future__ import annotations
from io import BytesIO
from typing import Optional

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

SCOPES = ["https://www.googleapis.com/auth/drive.file"]

def build_drive_service_from_info(service_account_info: dict):
    creds = service_account.Credentials.from_service_account_info(
        service_account_info, scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds)

def upload_png_bytes(
    drive_service,
    png_bytes: bytes,
    filename: str,
    folder_id: Optional[str] = None,
) -> str:
    file_metadata = {"name": filename, "mimeType": "image/png"}
    if folder_id:
        file_metadata["parents"] = [folder_id]

    media = MediaIoBaseUpload(BytesIO(png_bytes), mimetype="image/png", resumable=False)
    created = (
        drive_service.files()
        .create(body=file_metadata, media_body=media, fields="id")
        .execute()
    )
    return created["id"]
