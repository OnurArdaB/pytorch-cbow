<div class="cell code" data-execution_count="17" data-colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="yhkE4uU0IEkQ" data-outputId="b64a3f74-72de-40d6-af09-2a04f681d687">

``` python
from google.colab import drive
drive.mount("gdrive")
```

<div class="output stream stdout">

    Drive already mounted at gdrive; to attempt to forcibly remount, call drive.mount("gdrive", force_remount=True).

</div>

</div>

<div class="cell code" data-execution_count="18" data-colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="KNt_KPyGG6sh" data-outputId="95fe2e61-43ec-4720-be49-2b5bf3407049">

``` python
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import string
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = set(stopwords.words('turkish'))
from typing import List

import torch.nn
```

<div class="output stream stdout">

    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!

</div>

</div>

<div class="cell markdown" id="s4zxTB-dIVP6">

I am going to use a turkish news dataset.

</div>

<div class="cell code" data-execution_count="19" id="Fl6F90erIU3X">

``` python
train_df = pd.read_csv("/content/gdrive/MyDrive/CS-445/train.csv",index_col=[0])
```

</div>

<div class="cell code" data-execution_count="20" id="o5t-A8dwFmuO">

``` python
WINDOW_SIZE = 3  # 3 words to the left, 3 to the right
CORPUS = train_df["text"].to_list()[1000:1010]
```

</div>

<div class="cell code" data-execution_count="21" id="yTnse2k7HXAn">

``` python
def prune(text="",punc=string.punctuation.replace(".",""),stopwords=stopwords,lower=True)->List[str]:
  ''' 
    This function initially prunes a text from punctuation and stopwords and finally tokenizes a text.
    Parameters:
      text: (str) This is the text that will be processed.
      
      punc: (str) This is a string of punctuation characters. 
      Default is string.punctuation.
      
      stopwords: (List[str]) This is a list of turkish stop word strings.
      Default is nltk.stopwords.

    Returns:
      tmp: (List[str]) This is a list of tokenized and pruned text as string.
  '''
  for p in punc:
    text = text.replace(p," ")
  text = text.lower() if(lower) else text
  tokenized = word_tokenize(text)
  tmp = []
  for token in tokenized:
    if(token not in stopwords):
      tmp.append(token)
  return tmp
```

</div>

<div class="cell markdown" id="LydrkWNaGBq2">

First we should process corpus such that resulting output should be a
list of tokens with order.

</div>

<div class="cell code" data-execution_count="22" data-colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="fPi0J3z3J0ak" data-outputId="56bd48f2-c135-40b8-e449-ae4d73b416c6">

``` python
prune(text=CORPUS[0])
```

<div class="output execute_result" data-execution_count="22">

    ['suriye',
     'resmi',
     'haber',
     'ajansı',
     'sana',
     'amerika',
     'birleşik',
     'devletleri',
     'abd',
     'başkanı',
     'barack',
     'obama',
     'nın',
     'açıkladığı',
     'işi̇d',
     'stratejisini',
     'teröre',
     'destek',
     'olarak',
     'yorumladı',
     '.',
     'sana',
     'nın',
     'haberinde',
     'obama',
     'nın',
     'açıklamalarının',
     'suriye',
     'deki',
     'krizin',
     'siyasi',
     'çözümünün',
     'önünü',
     'tıkadığı',
     'savunuldu',
     '.',
     'haberde',
     'washington',
     'un',
     'bölgede',
     'özellikle',
     'suriye',
     'terörü',
     'destekleyen',
     'politikaları',
     'ülkedeki',
     'krize',
     'çözüm',
     'bulunması',
     'konusunda',
     'önemli',
     'engel',
     'teşkil',
     'ediyor',
     '.',
     'washington',
     'yönetimi',
     'bir',
     'yandan',
     'krizin',
     'çözülmesini',
     'istiyor',
     'bir',
     'yandan',
     'suriye',
     'mücadele',
     'eden',
     'teröristlere',
     'mal',
     'silah',
     'yardımı',
     'yapılması',
     'konusunda',
     'kararlar',
     'çıkartıyor',
     'denildi',
     '.',
     'haberde',
     'ayrıca',
     'washington',
     'çelişen',
     'politikaları',
     'konumuyla',
     'terörle',
     'mücadeleyi',
     'ciddiye',
     'almadığını',
     'gösteriyor',
     '.',
     'terör',
     'örgütlerinin',
     'bir',
     'kısmına',
     'savaş',
     'ilan',
     'edilirken',
     'abd',
     'kongresi',
     'nden',
     'işi̇d',
     'kadar',
     'suçlu',
     'olan',
     'suriyeli',
     'muhaliflerin',
     'silahlandırılması',
     'konusunda',
     'onay',
     'alınmaya',
     'çalışılıyor',
     'ifadeleri',
     'kullanıldı',
     '.',
     'suriye',
     'rejiminden',
     'dışişleri',
     'bakanlığından',
     'obama',
     'nın',
     'açıklamalarıyla',
     'ilgili',
     'resmi',
     'bir',
     'açıklama',
     'yapılmadı',
     '.',
     'obama',
     'işi̇d',
     'e',
     'karşı',
     'sistematik',
     'hava',
     'saldırıları',
     'düzenleyeceklerini',
     'belirtmiş',
     'kapsamda',
     'irak',
     'taki',
     'hava',
     'saldırılarını',
     'sadece',
     'amerikan',
     'personelini',
     'koruma',
     'insani',
     'yardım',
     'amaçlı',
     'olmanın',
     'ötesine',
     'taşıyacaklarını',
     'kaydetmişti',
     '.',
     'obama',
     'dolayısıyla',
     'irak',
     'güçleri',
     'hücumda',
     'olurken',
     'işi̇d',
     'hedeflerini',
     'vuracağız',
     '.',
     'dahası',
     'şuna',
     'açıklık',
     'getiriyorum',
     'ülkemizi',
     'tehdit',
     'eden',
     'teröristleri',
     'olurlarsa',
     'olsunlar',
     'ele',
     'geçireceğiz',
     '.',
     'suriye',
     'irak',
     'ta',
     'teröristlere',
     'karşı',
     'harekete',
     'geçmekte',
     'tereddüt',
     'etmeyeceğim',
     'anlamına',
     'geliyor',
     '.',
     'benim',
     'başkanlığımın',
     'ana',
     'ilkesi',
     'amerika',
     'yı',
     'tehdit',
     'ediyorsanız',
     'sığınacak',
     'güvenli',
     'bir',
     'yer',
     'bulamayacaksınız',
     'demişti',
     '.',
     'abd',
     'başkanı',
     'obama',
     'kongre',
     'ye',
     'suriyeli',
     'muhaliflere',
     'fazla',
     'ekipman',
     'sağlanması',
     'eğitim',
     'verilmesi',
     'kendisine',
     'ek',
     'yetki',
     'kaynak',
     'sağlaması',
     'çağrısında',
     'bulunmuştu',
     '.']

</div>

</div>

<div class="cell code" data-execution_count="23" id="tArUqieKGmwb">

``` python
PROCESSED_CORPUS = []
for text in CORPUS:
  PROCESSED_CORPUS.extend(prune(text=text))
```

</div>

<div class="cell code" data-execution_count="24" id="RIGbxmBeKsha">

``` python
vocabulary = set(PROCESSED_CORPUS)
vocabulary_size = len(vocabulary)
```

</div>

<div class="cell code" data-execution_count="25" data-colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="po2zU0n8KyFd" data-outputId="3a29cc01-36b4-4477-bf48-c2013da49510">

``` python
list(vocabulary)[:20]
```

<div class="output execute_result" data-execution_count="25">

    ['umarım',
     'favori',
     'birleşik',
     'gösteriyor',
     'geldi',
     'bugüne',
     'futbolcuya',
     'görüntülendi',
     'dedi',
     'madridli',
     'takım',
     'luciano',
     'erdemit',
     'hatasına',
     'karşısına',
     'küçümsemeyin',
     'harcanacağına',
     'anları',
     'gurur',
     'haberinde']

</div>

</div>

<div class="cell code" data-execution_count="26" data-colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="gsmRo6ooGAv6" data-outputId="a046ca3c-7bc5-46c5-e687-989256a3a33c">

``` python
word2index = {word: i for i, word in enumerate(vocabulary)}
data = [] # Will contain the focus and context words
for i in range(3, len(PROCESSED_CORPUS) - 3):
    context = [
               PROCESSED_CORPUS[i - 3], 
               PROCESSED_CORPUS[i - 2], 
               PROCESSED_CORPUS[i - 1],
               PROCESSED_CORPUS[i + 1], 
               PROCESSED_CORPUS[i + 2],
               PROCESSED_CORPUS[i + 3], 
               ]

    target = PROCESSED_CORPUS[i]
    data.append((context, target))

print("Context:" , data[0][0])
print("Target:" , data[0][1])
```

<div class="output stream stdout">

    Context: ['suriye', 'resmi', 'haber', 'sana', 'amerika', 'birleşik']
    Target: ajansı

</div>

</div>

<div class="cell markdown" id="Jw8kiOIkLgE3">

We are trying to obtain a special format where there exists a focues no

</div>

<div class="cell code" data-execution_count="27" data-colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="moQGEL9SM-ev" data-outputId="c0f60290-8964-42a5-ac66-ff0e79346f4a">

``` python
def make_context_vector(context, word2index):
    idxs = [word2index[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)

make_context_vector(data[0][0],word2index)
```

<div class="output execute_result" data-execution_count="27">

    tensor([1012,  891,  933, 1129,  868,    2])

</div>

</div>

<div class="cell code" data-execution_count="28" data-colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="H0sFjr4zM02b" data-outputId="920ea76b-5f43-4645-acb2-63267893be83">

``` python
data[0][0]
```

<div class="output execute_result" data-execution_count="28">

    ['suriye', 'resmi', 'haber', 'sana', 'amerika', 'birleşik']

</div>

</div>

<div class="cell code" data-execution_count="29" id="-oUduh_2LaUe">

``` python
class CBOW(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()

        #out: 1 x emdedding_dim
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = torch.nn.Linear(embedding_dim, 128)
        self.activation_function1 = torch.nn.ReLU()
        
        #out: 1 x vocab_size
        self.linear2 = torch.nn.Linear(128, vocab_size)
        self.activation_function2 = torch.nn.LogSoftmax(dim = -1)
        

    def forward(self, inputs):
        embeds = sum(self.embeddings(inputs)).view(1,-1)
        out = self.linear1(embeds)
        out = self.activation_function1(out)
        out = self.linear2(out)
        out = self.activation_function2(out)
        return out

    def word_emdedding(self, word):
        word = torch.tensor([word2index[word]])
        return self.embeddings(word).view(1,-1)
```

</div>

<div class="cell code" data-execution_count="30" id="myJZpOnhQ1wW">

``` python
model = CBOW(vocabulary_size, 2)

loss_function = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for epoch in range(50):
    total_loss = 0

    for context, target in data:
        context_vector = make_context_vector(context, word2index)  
        log_probs = model(context_vector)        
        total_loss += loss_function(log_probs, torch.tensor([word2index[target]]))

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

</div>

<div class="cell code" data-execution_count="31" data-colab="{&quot;height&quot;:952,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="t7P-q1KOoQ9C" data-outputId="d3246b88-f06c-484d-aa25-aa6ab639cd70">

``` python
plt.figure(figsize=(20, 20))
for word in list(word2index.keys())[:60]:
  coord = model.word_emdedding(word)
  plt.scatter(coord[0][0].item(), coord[0][1].item())
  plt.annotate(word, (coord[0][0].item(), coord[0][1].item()))
```

<div class="output display_data">

![](2557a0c29ea15885068adc93554fe176feb0184c.png)

</div>

</div>
