from typing import Callable
import pandas as pd
import xarray as xr


class NetcdfPool:
    def __init__(self, load: Callable[..., xr.Dataset]) -> None:
        self._idx = 0
        self._loader = load
        self._datasets = []

    def initialize(self, n: int) -> None:
        for _ in range(n):
            self._datasets.append(self._loader())

    def _inc_index(self) -> None:
        self._idx = (self._idx + 1) % len(self._datasets)

    def get_at(
        self,
        datetime: pd.Timestamp,
        lat: float,
        lon: float,
        method: str = "nearest",
        tolerance: float = 0.5,
    ) -> xr.Dataset:
        ds = self._datasets[self._idx]
        if lon < 0:
            lon += 360
        ret = ds.sel(
            latitude=lat,
            longitude=lon,
            time=datetime,
            method=method,
            tolerance=tolerance,
        )
        self._inc_index()
        return ret

    def close_all(self) -> None:
        for ds in self._datasets:
            ds.close()
