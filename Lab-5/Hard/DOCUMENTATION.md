## Вид векторного представления
### Структура вектора:

- Первые `PARAMS_SIZE` элементов содержат общие параметры сети (размеры входа, количество классов, глубина сети)
- Следующие `MAX_LAYERS * LAYER_VECTOR_SIZE` элементов содержат информацию о каждом слое
- Последние `MAX_CONNECTIONS * CONNECTION_VECTOR_SIZE` элементов содержат информацию о соединениях между слоями

### Общие параметры сети `(PARAMS_SIZE = 10`):

- `vector[0]` - ширина входного изображения (`input_shape[0]`)
- `vector[1]` - высота входного изображения (`input_shape[1]`)
- `vector[2]` - количество каналов входного изображения (`input_shape[2]`)
- `vector[3]` - количество выходных классов (`output_classes`)
- `vector[4]` - максимальное количество слоев (`MAX_LAYERS`)
- `vector[5]` - фактическое количество слоев (`len(self.layers)`)
- `vector[6]` - наличие остаточных соединений (`1 если есть, 0 если нет`)
- `vector[7]` - наличие Inception блоков (`1 если есть, 0 если нет`)
- `vector[8]` - глубина сети (`len(self.layers)`)
- `vector[9]` - пока не используется, просто для удобства оставлено))

### Представление слоя (`LAYER_VECTOR_SIZE = 12`):

- [0] - тип слоя (значение из LayerType)
- [1] - для `ConvLayer`: kernel_size[0], для `PoolLayer`: pool_size[0], для `InceptionLayer`: filters_1x1
- [2] - для `ConvLayer`: kernel_size[1], для `PoolLayer`: pool_size[1], для `InceptionLayer`: filters_3x3_reduce
- [3] - для `ConvLayer`: stride[0], для `PoolLayer`: stride[0], для `InceptionLayer`: filters_3x3
- [4] - для `ConvLayer`: stride[1], для `PoolLayer`: stride[1], для `InceptionLayer`: filters_5x5_reduce
- [5] - для `ConvLayer`: padding[0], для `PoolLayer`: padding[0], для `InceptionLayer`: filters_5x5
- [6] - для `ConvLayer`: padding[1], для `PoolLayer`: padding[1], для `InceptionLayer`: filters_pool_proj
- [7] - для `ConvLayer`: filters, для `FCLayer`: neurons
- [8] - для `ConvLayer/FCLayer/ActivationLayer`: тип активации (значение из ActivationType)
- [9] - для `PoolLayer`: тип пулинга (значение из `PoolType`), для `DropoutLayer`: rate
- [10] - для `ConvLayer`: dilation_rate
- [11] - пока не используется, просто для удобства оставлено))

### Представление соединения (`CONNECTION_VECTOR_SIZE = 3`):
Для каждого соединения выделяется вектор размером 3 элемента:

- [0] - индекс исходного слоя (from_layer)
- [1] - индекс целевого слоя (to_layer)
- [2] - тип соединения (значение из ConnectionType)

### Полное векторное представление:
Итоговый вектор представляет собой конкатенацию:

- Вектор общих параметров (10 элементов)
- Векторы для `MAX_LAYERS` слоев (каждый по 12 элементов)
- Векторы для `MAX_CONNECTIONS` соединений (каждый по 3 элемента)

### Пример векторного представления для слоев:
```
conv_layer = ConvLayer(
    kernel_size=(3, 3), 
    stride=(1, 1), 
    padding=(1, 1), 
    filters=64, 
    activation=ActivationType.RELU, 
    dilation_rate=1
)
[0, 3, 3, 1, 1, 1, 1, 64, 1, 0, 1, 0]


pool_layer = PoolLayer(
    pool_size=(2, 2), 
    stride=(2, 2), 
    padding=(0, 0), 
    pool_type=PoolType.MAX
)
[1, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0]


fc_layer = FCLayer(
    neurons=1024, 
    activation=ActivationType.SIGMOID
)
[2, 0, 0, 0, 0, 0, 0, 1024, 2, 0, 0, 0]


bn_layer = BNLayer()
[3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


dropout_layer = DropoutLayer(rate=0.5)
[4, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0]


inception_layer = InceptionLayer(
    filters_1x1=64,
    filters_3x3_reduce=96,
    filters_3x3=128,
    filters_5x5_reduce=16,
    filters_5x5=32,
    filters_pool_proj=32
)
[5, 64, 96, 128, 16, 32, 32, 0, 0, 0, 0, 0]
```
