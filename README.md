# Segmentation test case
#### Made by Valery Kasymov

В этом задании в качестве архитектуры была выбрана U-Net с небольшими модификациями.
Предобработка изображений-томограмм заключается в нормализации входных данных.
Маски были бинаризированы в соответсвтие с постановкой задачи.
Основной фреймворк: **PyTorch**.
В качестве треккера использовался **wandb**. По умолчанию треккинг выключен, для его активации необходимо задать параметр в функцию **solution** файла **main** аргумент **track = True**.

### Описание файлов
**myLoss**: реализация Dice лосс-функции
**myDataset**: релизация датасета с помощью PyTorch
**myUtils**: вспомогательные функции для чтения и предобработки данных
**myModel**: нейронная сеть архитектуры U-Net на PyTorch.
**myTrain**: тренировка модели. Включает в себя проход по эпохам и по батчам.
**main**: исполняемый файл. Содержит основные этапы чтения данных, инициализации модели, оптимизатора и функции ошибок. 

### Запуск
В случае запуска из консоли, можно запустить только файл main.py. В таком случае не будет треккинга, а за директорию будет выбрана та, в которой исполняемый файл.
В таком случае необходимо, чтобы папка **subset** лежала вместе с данным файлом в одной директории.
```sh
python main.py
```
Для запуска из Jupyter Notebook необходимо сделать 

```sh
from main import solution
solution(dir, track)
#dir - директория, в который лежит папка subset
#track - активация wandb
```

### Вывод
Модель сохраняется на каждой итерации, в которой метрика Dice лучше, чем лучшая за прошлые эпохи в файл **outputs.pickle**.
Зависимость Dice от эпох записывается в файл **DiceLoss.png**

## Лицензия
MIT
