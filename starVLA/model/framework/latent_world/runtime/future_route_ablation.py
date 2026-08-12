from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class FutureRouteAblation:
    future_off: bool
    auxiliary_off: bool
    conditioning_off: bool

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str],
        *,
        future_prediction: bool,
        dual_route: bool,
    ) -> "FutureRouteAblation":
        routes = cls(
            future_off=environment.get("LAWAM_FUTURE_OFF") == "1",
            auxiliary_off=environment.get("LAWAM_AUXILIARY_OFF") == "1",
            conditioning_off=environment.get("LAWAM_CONDITIONING_OFF") == "1",
        )
        routes.validate(
            future_prediction=future_prediction,
            dual_route=dual_route,
        )
        return routes

    def validate(self, *, future_prediction: bool, dual_route: bool) -> None:
        if self.future_off and not future_prediction:
            raise ValueError(
                "LAWAM_FUTURE_OFF requires future_prediction=true so parameter and "
                "trainable trees remain matched."
            )
        if self.future_off and dual_route:
            raise ValueError(
                "LAWAM_FUTURE_OFF is defined only for the matched single future route."
            )
        if (self.auxiliary_off or self.conditioning_off) and not future_prediction:
            raise ValueError(
                "LAWAM_AUXILIARY_OFF and LAWAM_CONDITIONING_OFF require "
                "future_prediction=true."
            )
        if self.future_off and (self.auxiliary_off or self.conditioning_off):
            raise ValueError(
                "LAWAM_FUTURE_OFF already disables both future routes and cannot be "
                "combined with single-route ablations."
            )

    @property
    def auxiliary_enabled(self) -> bool:
        return not self.future_off and not self.auxiliary_off

    @property
    def conditioning_enabled(self) -> bool:
        return not self.future_off and not self.conditioning_off
