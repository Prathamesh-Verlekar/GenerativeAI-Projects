from __future__ import annotations

import json
from typing import List, Tuple

import gspread
from google.oauth2.service_account import Credentials


SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


class GoogleSheetsClient:
    def __init__(self, service_account_json: str):
        info = json.loads(service_account_json)
        creds = Credentials.from_service_account_info(info, scopes=SCOPES)
        self._client = gspread.authorize(creds)

    def _get_worksheet(self, spreadsheet_id: str, worksheet_title: str | None):
        spreadsheet = self._client.open_by_key(spreadsheet_id)
        if worksheet_title:
            return spreadsheet.worksheet(worksheet_title)
        return spreadsheet.sheet1

    def fetch_range(self, spreadsheet_id: str, range_a1: str, worksheet_title: str | None) -> Tuple[str, List[List[str]]]:
        worksheet = self._get_worksheet(spreadsheet_id, worksheet_title)
        return worksheet.title, worksheet.get(range_a1)

    def append_row(self, spreadsheet_id: str, row: List[str], worksheet_title: str | None) -> Tuple[str, str]:
        worksheet = self._get_worksheet(spreadsheet_id, worksheet_title)
        response = worksheet.append_row(row, value_input_option="USER_ENTERED")
        updated_range = response.get("updates", {}).get("updatedRange", "") if isinstance(response, dict) else ""
        return worksheet.title, updated_range

    def update_range(
        self, spreadsheet_id: str, range_a1: str, values: List[List[str]], worksheet_title: str | None
    ) -> Tuple[str, str]:
        worksheet = self._get_worksheet(spreadsheet_id, worksheet_title)
        response = worksheet.update(range_a1, values, value_input_option="USER_ENTERED")
        updated_range = response.get("updatedRange", "") if isinstance(response, dict) else ""
        return worksheet.title, updated_range

    def head_rows(self, spreadsheet_id: str, worksheet_title: str | None, limit: int) -> Tuple[str, List[List[str]]]:
        worksheet = self._get_worksheet(spreadsheet_id, worksheet_title)
        values = worksheet.get_all_values()
        return worksheet.title, values[:limit]
