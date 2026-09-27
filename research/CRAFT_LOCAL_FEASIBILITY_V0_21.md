# CRAFT v0.21 reduced-context local feasibility result

- Structure: `structure_001` (medium); completed: `False`.
- Turns: 2; final progress: 0.131; stopping: `max_turns_reached`.
- Model calls: 8; prompt tokens: 15218; generated tokens: 739.
- Summed llama.cpp service time: 38.571 s; mean prompt/generation throughput: 1369.23/26.05 tokens/s.
- Parser residuals: 6/6 messages retained square-bracket wrappers; 3/6 retained a format tag.

| Turn | Generated | Parsed | Builder received | Progress | Selected move | Wire tokens incl. role labels |
|---:|---:|---:|---:|---:|---|---:|
| 1 | 3 | 3 | 3 | 0.068 | `{'action': 'place', 'block': 'ys', 'position': '(0,0)', 'layer': 0, 'span_to': None}` | 94 |
| 2 | 3 | 3 | 3 | 0.131 | `{'action': 'place', 'block': 'bl', 'position': '(1,0)', 'layer': 0, 'span_to': '(2,0)'}` | 92 |

## Public messages

### Turn 1

- **D1** (parsed=True, delivered=True, tokens=38, leftover format tags=0): [My bottom left has a yellow block, but the middle and right need two large blue ones. Place those large blue blocks across the middle and right of my bottom layer.]
- **D2** (parsed=True, delivered=True, tokens=41, leftover format tags=1): [So the bottom layer near me needs a yellow block, then a green one, and then an orange one. Put a yellow small block in my bottom left corner.]
</message>
- **D3** (parsed=True, delivered=True, tokens=14, leftover format tags=0): [Put an orange small block in my bottom left.]

### Turn 2

- **D1** (parsed=True, delivered=True, tokens=20, leftover format tags=0): [My bottom middle and right need two large blue blocks. Place those there now.]
- **D2** (parsed=True, delivered=True, tokens=44, leftover format tags=1): [So the bottom layer near me needs a yellow block, then a green one, and then an orange one. Put a green small block next to the yellow in my bottom middle.]
</message>
- **D3** (parsed=True, delivered=True, tokens=28, leftover format tags=1): [The middle and right of my bottom layer are missing green blocks. Add two large green ones there.]
</message>

## Interpretation

All generated, parsed, and Builder-delivered counts are kept separate. This run establishes only that the reduced-context local configuration completed the small CRAFT slice. Progress remained partial. It does not compare languages, estimate generalization, or establish a communication-efficiency advantage.
