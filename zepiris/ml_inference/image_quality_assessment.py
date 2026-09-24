"""Image quality assessment combining NSFW, spoof, and blur detection."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from zepiris.ml_inference.blur_detection import BlurDetectionService
from zepiris.ml_inference.nsfw_detection import NSFWDetectionService
from zepiris.ml_inference.spoof_detection import SpoofDetectionService
from zepiris.schemas.ml_inference import ImageQualityAssessmentResult

logger = logging.getLogger(__name__)


class ImageQualityAssessmentService:
    """Image quality assessment service combining three quality checks.

    Runs NSFW detection, spoof detection, and blur detection on an image
    in parallel using a thread pool. Accepts pre-loaded service instances
    to avoid redundant model loading.

    Blur is optional. NSFW and spoof are safety gates and a missing model there
    means the service cannot answer at all, but a missing blur model only costs
    one sharpness number — so the assessment runs without it and reports
    ``blur=None`` instead of failing every request.
    """

    def __init__(
        self,
        nsfw_service: NSFWDetectionService,
        spoof_service: SpoofDetectionService,
        blur_service: BlurDetectionService | None,
    ) -> None:
        self.nsfw_service = nsfw_service
        self.spoof_service = spoof_service
        self.blur_service = blur_service
        self._executor = ThreadPoolExecutor(max_workers=3)

    def assess(self, image_rgb: np.ndarray) -> ImageQualityAssessmentResult:
        """Run image quality assessment on all available checks in parallel.

        Args:
            image_rgb: Input image in RGB format, shape (H, W, 3), dtype uint8

        Returns:
            ImageQualityAssessmentResult: Aggregated quality assessment result.
                passed=True if every check that ran passed (is_safe AND is_live,
                AND is_sharp when the blur model is loaded), otherwise False.
                blur is None when the blur model is unavailable or its inference
                raised — the other checks are still reported.
        """
        nsfw_future = self._executor.submit(self.nsfw_service.forward, image_rgb)
        spoof_future = self._executor.submit(self.spoof_service.forward, image_rgb)
        blur_future = (
            self._executor.submit(self.blur_service.forward, image_rgb)
            if self.blur_service is not None
            else None
        )

        nsfw_result = nsfw_future.result()
        spoof_result = spoof_future.result()

        blur_result = None
        if blur_future is not None:
            try:
                blur_result = blur_future.result()
            except Exception:
                # A blur failure must not sink an assessment whose safety gates
                # both answered; skip the check and report it as absent.
                logger.exception("Blur detection failed; continuing without it")

        passed = nsfw_result.is_safe and spoof_result.is_live
        if blur_result is not None:
            passed = passed and blur_result.is_sharp

        return ImageQualityAssessmentResult(
            passed=passed,
            nsfw=nsfw_result,
            spoof=spoof_result,
            blur=blur_result,
        )
