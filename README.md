
# ELTE Poetry Corpus

> Originally from: Horváth Péter – Kundráth Péter – Indig Balázs – Fellegi Zsófia – Szlávich Eszter – Bajzát Tímea Borbála – Sárközi-Lindner Zsófia – Vida Bence – Karabulut Aslihan – Timári Mária – Palkó Gábor 2022. ELTE Verskorpusz – a magyar kanonikus költészet gépileg annotált adatbázisa. In: Berend Gábor – Gosztolya Gábor – Vincze Veronika (szerk.): XVIII. Magyar Számítógépes Nyelvészeti Konferencia. Szeged: Szegedi Tudományegyetem TTIK, Informatikai Intézet. 375–388.

# What I looked at

### BERTopic

I experimented with BERTopic, which I have found to have worked slightly worse, or about the same (with more computation power needed) as classic LDA. 

While this is true, based on coherence score alone, it is important  to note, that I intentionally challenged BERTopic, with a task, where attention is barely usable due to stylistic use of language; and sentence transformers could not have been used due to similar stylistic problems. I embedded the poems with [@oroszgy's HuSpacy](https://github.com/huspacy/huspacy) 300d floret model, trained on news, and lemmatized with the same model. Archaic text is yet to be implemented in language models.

### Multinomial Text Classification with Spacy

I trained a classifier model, to recognize poets - with varying results. 

=========================== Textcat F (per label) ===========================

                                    P       R       F
                    Ady            81.20   89.62   85.20
                    AranyJ         72.09   81.58   76.54
                    Babits         68.29   62.22   65.12
                    Balassi       100.00   77.78   87.50
                    Csokonai       94.44   85.00   89.47
                    Jozsef         81.13   74.14   77.48
                    Karinthy        0.00    0.00    0.00
                    Kolcsey       100.00   83.33   90.91
                    Kosztolanyi    72.86   82.26   77.27
                    Madach        100.00   66.67   80.00
                    Petofi         90.54   84.81   87.58
                    Radnoti        84.62   64.71   73.33
                    Vorosmarty     66.25   92.98   77.37


======================== Textcat ROC AUC (per label) ========================

                                ROC AUC
                    Ady              0.97
                    AranyJ           0.98
                    Babits           0.94
                    Balassi          1.00
                    Csokonai         0.98
                    Jozsef           0.97
                    Karinthy         0.62
                    Kolcsey          0.96
                    Kosztolanyi      0.98
                    Madach           0.98
                    Petofi           0.99
                    Radnoti          0.99
                    Vorosmarty       0.98