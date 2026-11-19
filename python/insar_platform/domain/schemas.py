"""Validated control-plane and compute request schemas."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class AlertRules(BaseModel):
    threshold_mm: float = Field(25.0, gt=0, description="Flag high-coherence motion exceeding +/- mm")
    min_coherence: float = Field(0.65, ge=0.0, le=1.0)


class CreateJobRequest(BaseModel):
    monitor_name: str = Field(..., examples=["yorkshire_trunk_main_north"])
    granule_id: Optional[str] = Field(default=None, description="Provider granule identifier used for pass idempotency")
    engine_version: str = Field(default="0.5.0")
    idempotency_key: Optional[str] = None
    slc1_path: str = Field(..., description="Reference OPERA CSLC-S1 .h5, NISAR GSLC, complex GeoTIFF, or worker-readable s3:// URI")
    slc2_path: str = Field(..., description="Secondary OPERA CSLC-S1 .h5, NISAR GSLC, complex GeoTIFF, or worker-readable s3:// URI")
    subdataset: str = "/data/VV"
    wavelength_mm: float = 55.4658
    stride_x: int = 6
    stride_y: int = 3
    ref_lonlat: Optional[tuple[float, float]] = None
    ref_radius_m: float = Field(1000.0, gt=0)
    ref_px: Optional[tuple[int, int, int]] = None
    alert_rules: AlertRules = Field(default_factory=AlertRules)
    webhook_url: Optional[str] = None


class ClosureTripletRequest(BaseModel):
    disp_a_path: str
    disp_b_path: str
    mode: Literal["add_sub", "sum_sub"] = "add_sub"


class StackUpdateRequest(BaseModel):
    monitor_name: Optional[str] = None
    engine_version: str = "0.5.0"
    idempotency_key: Optional[str] = None
    source_job_id: Optional[str] = None
    disp_path: Optional[str] = None
    coh_path: Optional[str] = None
    pair_job: Optional[CreateJobRequest] = None
    dt_days: Optional[float] = Field(default=None, gt=0)
    t0_days: Optional[float] = None
    t1_days: Optional[float] = None
    ramp_order: int = Field(1, ge=-1, le=2)
    min_coh: float = Field(0.35, ge=0.0, le=1.0)
    dem_path: Optional[str] = None
    ref_lonlat: Optional[tuple[float, float]] = None
    ref_radius_m: float = Field(1000.0, gt=0)
    ref_px: Optional[tuple[int, int, int]] = None
    ref_region_path: Optional[str] = None
    deform_mask_path: Optional[str] = None
    prior_vel_path: Optional[str] = None
    ramp_deform_vel_mm_yr: float = Field(25.0, ge=0.0)
    ramp_mask_sigma: float = Field(2.5, ge=0.0)
    ramp_mask_dilate_px: int = Field(2, ge=0, le=25)
    closure_triplets: list[ClosureTripletRequest] = Field(default_factory=list)
    closure_thresh_mm: float = Field(10.0, gt=0)
    closure_max_bad_frac: float = Field(0.35, ge=0.0, le=1.0)
    closure_mode: Literal["consensus", "strict"] = "consensus"
    redundant_pair: bool = False
    pair_weight_scale: float = Field(1.0, gt=0)
    corridors_path: Optional[str] = None
    buffer_m: float = Field(100.0, ge=0.0)
    vel_alert_mm_yr: float = Field(15.0, gt=0)
    accel_alert_mm_yr2: float = Field(25.0, gt=0)
    min_alert_coh: float = Field(0.55, ge=0.0, le=1.0)
    min_alert_snr: float = Field(2.5, ge=0.0)
    min_alert_span_days: float = Field(60.0, ge=0.0)
    webhook_url: Optional[str] = None


class SyncMonitorRequest(BaseModel):
    """Optional overrides for a configured monitor catalog sync."""

    engine_version: str = "0.5.0"
    idempotency_key: Optional[str] = None
    webhook_url: Optional[str] = None
