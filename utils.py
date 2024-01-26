from enum import Enum
from typing import Tuple

class VisiblePage(Enum):
  LEFT = 1
  RIGHT = 2

class ProcessedPage:
  height: int = 0
  width: int = 0
  left_line: Tuple[int, int, int, int] = None
  right_line: Tuple[int, int, int, int] = None
  page_coordinates = None

  def __init__(self, left_line: Tuple[int, int, int, int], right_line: Tuple[int, int, int, int]):
      self.left_line = left_line
      self.right_line = right_line
