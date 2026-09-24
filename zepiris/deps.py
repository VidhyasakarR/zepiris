from typing import Annotated

from fastapi import Depends, Request

from zepiris.config import Settings, get_settings
from zepiris.services.embedding import FaceEmbeddingProvider
from zepiris.services.iqa import MLInferenceIQAService
from zepiris.services.learning import AdaptiveThresholdLearner
from zepiris.services.matching import FaceMatcher
from zepiris.services.ml_client import AsyncMLInferenceClient
from zepiris.services.s3_fetcher import S3ImageFetcher


def settings_dep() -> Settings:
    return get_settings()


def iqa_dep(request: Request) -> MLInferenceIQAService:
    return request.app.state.iqa


def embedding_dep(request: Request) -> FaceEmbeddingProvider:
    return request.app.state.embedding


def matcher_dep(request: Request) -> FaceMatcher:
    return request.app.state.matcher


def s3_fetcher_dep(request: Request) -> S3ImageFetcher:
    return request.app.state.s3_fetcher


def ml_async_dep(request: Request) -> AsyncMLInferenceClient:
    return request.app.state.ml_async


def learner_dep(request: Request) -> AdaptiveThresholdLearner:
    return request.app.state.learner


SettingsDep = Annotated[Settings, Depends(settings_dep)]
IQADep = Annotated[MLInferenceIQAService, Depends(iqa_dep)]
EmbeddingDep = Annotated[FaceEmbeddingProvider, Depends(embedding_dep)]
MatcherDep = Annotated[FaceMatcher, Depends(matcher_dep)]
S3FetcherDep = Annotated[S3ImageFetcher, Depends(s3_fetcher_dep)]
LearnerDep = Annotated[AdaptiveThresholdLearner, Depends(learner_dep)]
MLAsyncDep = Annotated[AsyncMLInferenceClient, Depends(ml_async_dep)]
