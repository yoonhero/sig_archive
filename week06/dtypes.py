from typing import TypeAlias

# type Action = int only support on 3.12 or newer
Action: TypeAlias = int # 0~4171
Actions: TypeAlias = list[Action]
Token: TypeAlias = str # e1, <me>, <board_start> ...
Tokens: TypeAlias = list[Token]
TokenIndex: TypeAlias = int # 0~4295
Vector: TypeAlias = list[TokenIndex]
UCI: TypeAlias = str
UCIs: TypeAlias = list[UCI]
SAN: TypeAlias = str