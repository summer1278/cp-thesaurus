# cp-thesaurus
core-peri thesaurus for feature expansion

Publication:
```
@inproceedings{cui2018solving,
  title={Solving Feature Sparseness in Text Classification using Core-Periphery Decomposition},
  author={Cui, Xia and Kojaku, Sadamori and Masuda, Naoki and Bollegala, Danushka},
  booktitle={Proceedings of the Seventh Joint Conference on Lexical and Computational Semantics},
  pages={255--264},
  year={2018}
}
```


## preprocess.py
- ```word_ids_generator()```: generate ```word_ids``` in ```../data/```
- ```compute_links()```: generate and store ```ppmi_links``` in ```../data/``` (requires: ```word_ids```, ```ppmi.values```)
- ```compute_freq_coreness(domain)```: in-domain frequency as coreness for train and test data (requires: ```train```, ```test``` in label-sentence format, and ```word_ids```)
- ```compute_ppmi_coreness(domain)```: same as above but generate ppmi as coreness (requires: ```train```, ```test``` and ```word_ids```)
- ```convert_cp_nonoverlap(domain,method)```: convert ```km``` results to ```core coreness peri1,score1,peri2,score2,..```, ids replaced with words
- ```convert_cp_overlap(domain,method)```: convert ```km_overlap``` results to ```core coreness peri1,score1,peri2,score2,..```, ids replaced with words
- ```sort_peris(peris_list,core,h)``` and ```get_h()```: subfunctions supporting format convertion in ppmi decsending order


### Directories
Pre-computed requirements for ```convert_cp_overlap()``` or ```convert_cp_nonoverlap()``` :
- ```../data/ppmi.values```
- ```../data/word_ids```: generated from ```word_ids_generator()```
- ```../data/domain/result_method_overlap.dat```: generated from ```km_overlap```
- ```../data/domain/result_method_nonoverlap.dat```: generated from ```km```

Outputs:
- ```../data/domain/cpwords_method_overlap.dat``` : cp_overlap
- ```../data/domain/cpwords_method_nonoverlap.dat``` : cp_nonoverlap

#### current root directory for files in NLP1 server
- source code: ```xiacui2@nlp1: ~/python/cp-thesaurus/src```
- datasets: ```xiacui2@nlp1: ~/python/cp-thesaurus/data```

### TO RUN
packages: ```numpy```, ```math```

specify ```domain``` and ```method```, then uncomment functions in ```main()``` (e.g. ```domain = "TR", method = "ppmi"```)

```python preprocess.py``` 

## runner.py
script for automatically running step by step
1. compute ppmi values
2. km (nonoverlap) or km_overlap
3. convert cp result to words version and add the ppmi values to them
4. use 3 to expand the features and calculate the experimental results

### Required Files
- kmcpp code: ```../../kmcpp```
- word_ids:```../data/word_ids``` (if not exists, you can generate from ```preprocess.py```)
- ppmi_links:```../data/ppmi_links.dat``` (if not exists, you can generate from ```preprocess.py```)
- train and test data: ```../data/domain/train``` and ```../data/domain/test```

### TO RUN
from ```~/python/cp-thesaurus/src```
- USAGE: ```python runner.py <option:overlap or nonoverlap> <dataset or domain>```
- example: ```python runner.py overlap B-D```
- example: ```python runner.py nonoverlap TR```
