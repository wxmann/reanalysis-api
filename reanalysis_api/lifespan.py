from contextlib import asynccontextmanager

from fastapi import FastAPI

from reanalysis_api.datasources.era5 import arco_era5_isobaric, arco_era5_native
from reanalysis_api.pool import NetcdfPool


era5_native_pool = NetcdfPool(arco_era5_native)
era5_isobaric_pool = NetcdfPool(arco_era5_isobaric)


@asynccontextmanager
async def lifespan(app: FastAPI):
    era5_native_pool.initialize(3)
    era5_isobaric_pool.initialize(3)
    yield
    era5_isobaric_pool.close_all()
    era5_native_pool.close_all()
