from typing import TypeAlias

# type Action = int only support on 3.12 or newer
Action: TypeAlias = int
Actions: TypeAlias = list[Action]
Token: TypeAlias = int
Tokens: TypeAlias = list[Token]
UCI: TypeAlias = str
UCIs: TypeAlias = list[UCI]
SAN: TypeAlias = str