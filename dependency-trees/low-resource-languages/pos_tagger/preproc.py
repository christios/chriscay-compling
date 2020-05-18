#!/usr/bin/env python3

import re
import pdb

# with open("paste") as fh:
with open("MADAR.corpus26.MSA") as fh:
	data = fh.read()
	data = re.sub(r' ([^\d\w])', r'\1', data)
	data = re.sub(r',', r'،', data)

# with open("MSA-Beirut", 'w') as fh:
with open("MADAR.corpus26.MSA-preproc", 'w') as fh:
	# fh.write(re.sub(r'\t', r' ||| ', data))
	fh.write(data)