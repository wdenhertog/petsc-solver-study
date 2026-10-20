from enum import IntEnum


class CellTag(IntEnum):
    HEALTHY = 1
    WOUND = 2


class FacetTag(IntEnum):
    OUTER_BOUNDARY = 1
    WOUND_BOUNDARY = 2
