# Введение в ML system design

[![ML System design](http://img.youtube.com/vi/h5pPNDz-qUQ/0.jpg)](http://www.youtube.com/watch?v=h5pPNDz-qUQ "ML System design")

[Jupyter Notebook](../src/jupyter_notebooks/lecture_06_ml_system_design_intro.ipynb)

[slides](https://docs.google.com/presentation/d/1IFObAoJ3B-MqxE5Vo0GxCbQvzvPxDAjoVDoPva9oV6w/edit?usp=sharing)


## Шаг 1: среда разработки

Скачать дамп [messages.db](https://drive.google.com/file/d/1Ej6pV_GAXFDGxMk45Dntn2pnSlnn6IRs/view?usp=sharing) и положить в директорию [data](./data)

Запустить сборку докер-контейнера для разработки

```shell
make build
```

## Шаг 2

Этап EDA (Exploratory Data Analysis) - запускаем Jupyter чтобы "покопаться" в данных

```shell
make notebook
```

Открываем браузер по ссылке [localhost:8888](http://localhost:8888/)

После EDA сохраняем файл [scored_corpus.csv](https://drive.google.com/file/d/1lRpQOCwxwt0JAU9wDUOvhJ3CaYZMYFO_/view?usp=share_link) в директорию `data` (либо можно скачать из google drive по ссылке)

## Шаг 3: Разворачиваем LabelStudio

На этом шаге нужно разметить выборку в LabelStudio. Запускаем интерфейс командой

```shell
make labelstudio
```

Далее

* на вкладке `Sign up` вводим любой логин и пароль
* создаём проект и загружаем датасет на вкладке **Data import**
* 
* на вкладке `Labeling`  `Ordered By Time`
* Label all tasks
* размечаем на positive/negative
* когда датасет размечен - нажимаем "export"

Сохраняем размеченный датасет в файл `labeled_messages.csv`

# Шаг 5: Эксплуатация модели

Скачиваем размеченный датасет [по ссылке](https://drive.google.com/file/d/1MrxsEbeeJnIMdjL_GjsYysdKADyF5EQo/view?usp=sharing)

добавляем модель с микросервис

* [__main__](../src/train.py) - обучить модель с качеством f1 больше 0.86106
    *  Bert для фичей
    * более сложная модель (бустинг?)
* [pridict_labell](../src/service.py) - загрузить модель в сервис и реализовать API
    * `feed`
    * `/messages/<string:identifier>'`
* прислать PR в репозиторий
