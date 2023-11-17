from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import (
    ClassVar as _ClassVar,
    Iterable as _Iterable,
    Mapping as _Mapping,
    Optional as _Optional,
    Union as _Union,
)

DESCRIPTOR: _descriptor.FileDescriptor

class PredictRequest(_message.Message):
    __slots__ = ["tasks"]
    TASKS_FIELD_NUMBER: _ClassVar[int]
    tasks: _containers.RepeatedCompositeFieldContainer[TaskInfo]
    def __init__(
        self, tasks: _Optional[_Iterable[_Union[TaskInfo, _Mapping]]] = ...
    ) -> None: ...

class PredictResponse(_message.Message):
    __slots__ = ["prediction_results"]
    PREDICTION_RESULTS_FIELD_NUMBER: _ClassVar[int]
    prediction_results: _containers.RepeatedScalarFieldContainer[float]
    def __init__(
        self, prediction_results: _Optional[_Iterable[float]] = ...
    ) -> None: ...

class TaskInfo(_message.Message):
    __slots__ = [
        "due_date",
        "estimated_hours",
        "completed_on",
        "current_labor_hours",
        "category_name",
        "site_name",
        "created_at",
        "is_completed_on_time",
    ]
    DUE_DATE_FIELD_NUMBER: _ClassVar[int]
    ESTIMATED_HOURS_FIELD_NUMBER: _ClassVar[int]
    COMPLETED_ON_FIELD_NUMBER: _ClassVar[int]
    CURRENT_LABOR_HOURS_FIELD_NUMBER: _ClassVar[int]
    CATEGORY_NAME_FIELD_NUMBER: _ClassVar[int]
    SITE_NAME_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    IS_COMPLETED_ON_TIME_FIELD_NUMBER: _ClassVar[int]
    due_date: str
    estimated_hours: float
    completed_on: str
    current_labor_hours: float
    category_name: str
    site_name: str
    created_at: str
    is_completed_on_time: bool
    def __init__(
        self,
        due_date: _Optional[str] = ...,
        estimated_hours: _Optional[float] = ...,
        completed_on: _Optional[str] = ...,
        current_labor_hours: _Optional[float] = ...,
        category_name: _Optional[str] = ...,
        site_name: _Optional[str] = ...,
        created_at: _Optional[str] = ...,
        is_completed_on_time: bool = ...,
    ) -> None: ...
