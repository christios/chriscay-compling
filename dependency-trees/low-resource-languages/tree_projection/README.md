# Tree Projection

Source Language: `Italian`  
Target Language: `French`  
Alignment Type: `Union`

This is a rough implementation of Hwa et al., (2005), which aims at transferring dependency relations from a source langauge to a target language. The former is supposed to be resource-rich and the latter resource-poor. In my implementation, `fast align` alignments are leveraged and divided into the difference types of alignments (one-to-one, many-to-one, etc.) and are then processed according to this taxonomy. The implementation is still ongoing.

I am using treebank samples from the Universal Dependencies 2.5 set of treebanks (CoNLL-U format).

Run python command: `python3 project.py it_pud-ud-test.conllu fr_pud-ud-test.conllu it-fr.union`  

If you add the `-t` flag to the python command, then 20 alignments and their statistics will
be displayed. Else, the sentences are processed and printed for evaluation, and global statistics are printed.

## References
Hwa, Rebecca & Resnik, Ps & Weinberg, Amy & Cabezas, Clara & Kolak, Okan. (2005). Bootstrapping parsers via syntactic projection across parallel texts. Natural Language Engineering. 11. 311-325. 10.1017/S1351324905003840. 