I'm supposed to finetune Vietnamese news summarization for [Falconsai/text_summarization](https://huggingface.co/Falconsai/text_summarization)

## Download Vietnews data

        wget 'https://github.com/ThanhChinhBK/vietnews/archive/master.zip'
        unzip 'master.zip'
        

## Datasets
- [Wikilingua](https://github.com/esdurmus/Wikilingua)
- [Vietnews](https://github.com/ThanhChinhBK/vietnews)
- [Pho_NER](https://github.com/VinAIResearch/PhoNER_COVID19)

## Model
The model can be found here: https://huggingface.co/chibao24/vietnamese_mt5_summary_model

## Performance
5 text - chibao24/vietnamese_mt5_summary_model
- use_cahe=True
- CPU w/o batch: 10.352751731872559
- GPU w/o batch: 5.818488597869873
- CPU: 4.4028120040893555
- GPU: 2.0009677410125732
- GPU + bfloat16: 2.177988052368164 "same generated text"
- GPU + INT8: 7.794734239578247 "same generated text"

10 text - chibao24/vietnamese_mt5_summary_model
- GPU + bfloat16: 2.879676580429077 "same generated text" 

10 text - VietAI/vit5-large-vietnews-summarization
- GPU: 6.69340443611145
- GPU + bfloat16: 2.2307209968566895
