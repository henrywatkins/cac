# cac

Characterizing arbitrary collections
-----

This repository contains a means of characterising the identity of an arbitrary collection of items in terms of their constituent terms.

e.g. if we have three lists of strings ['apple','banana','pear'], ['orange','apple','pear'], ['pear','banana','lemon'], each beloinging to its own class, what are the characteristic terms of those classes?

## Installation

Clone the repository and 
```
pip install -e .
```

## Usage

```python
from cac import *

group_ids = [1, 2, 3, 1, 1]
data = [['apple', 'banana', 'cherry'], ['dog', 'elephant', 'fox'], ['green', 'yellow', 'blue'], ['car', 'bus', 'train'], ['red', 'orange', 'purple']]
top_k = 1

result = cfeatures(group_ids, data, top_k, show_values=True)
``` 