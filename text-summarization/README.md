# Text Summarization

Text summarization module which performs LSA to produce sentences extracted from the text which represent it the most. LSA uses Singular Value Decomposition (SVD), and then uses the VT matrix produced by SVD (the columns of which represent the sentences) to extract sentences from the text.

<p align="center">
  <img width="70%" height="70%" src="./assets/svd.png">
</p>

Note: Code is based on [4].

## Usage

- If you want to analyze your own document:

    `python3 lsa.py -i <input-file> --api-key <g3 api-key> [-e]`

- If you just want to test the module using an existing g3.Analysis file (an example g3.Analysis object is present in this repository called `analysis.json`):
    
    `python3 lsa.py -t <analysis.json> [-e]`

    *Note: If you want the analysis to be based on entities rather than tokens, add the `-e` flag.*

## TODO

- Tweak the parameters which decide which sentences will be disregarded (it heavily affects the results)
- The `-e` mode is yielding bad results; consider changing the way indexing is done (e.g, take POS and relations into account in addition to entity information)
- Change the way the score of a sentence if calculated based on the Cross Method from [3].
- Perform testing using some dataset and the ROUGE score

## References

[1]M. Berry, S. Dumais and G. O’Brien, "Using Linear Algebra for Intelligent Information Retrieval", SIAM Review, vol. 37, no. 4, pp. 573-595, 1995.

[2]J. Steinberger and K. Ježek, "Using Latent Semantic Analysis in Text Summarization and Summary Evaluation", 2004. Available: http://www.kiv.zcu.cz/~jstein/publikace/isim2004.pdf.

[3]M. Ozsoy, I. Cicekli and F. Alpaslan, "Text Summarization of Turkish Texts using Latent Semantic Analysis", 2010. Available: https://www.aclweb.org/anthology/C10-1098.pdf. [Accessed 22 September 2020].

[4]"iamprem/summarizer", GitHub, 2020. [Online]. Available: https://github.com/iamprem/summarizer.