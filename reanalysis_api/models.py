from datetime import datetime
from pydantic import BaseModel


class ProfileData(BaseModel):
    pressure: list[float | None]
    height: list[float | None]
    temperature: list[float | None]
    dewpoint: list[float | None]
    u_wind: list[float | None]
    v_wind: list[float | None]
    omega: list[float | None]


class ProfileUnits(BaseModel):
    pressure: str
    height: str
    temperature: str
    dewpoint: str
    u_wind: str
    v_wind: str
    omega: str


class ProfileMetadata(BaseModel):
    lat: float
    lon: float
    timestamp: datetime


class ProfileQuery(BaseModel):
    lat: float
    lon: float
    timestamp: datetime


class Profile(ProfileMetadata):
    data: ProfileData
    units: ProfileUnits


# class ProfileResponse(Profile):
#     _query: ProfileQuery
