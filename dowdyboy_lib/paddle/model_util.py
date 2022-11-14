import paddle


@paddle.no_grad()
def print_model(model, input_size, custom_ops=None, ):
    def number_format(x):
        return f'{round(x / 1.e6, 2)}M, {round(x / 1.e9, 2)}G'
    summary_res = paddle.summary(model, input_size, )
    flops_res = paddle.flops(model, input_size, custom_ops=custom_ops, )
    total_params = summary_res['total_params']
    trainable_params = summary_res['trainable_params']
    flops_count = flops_res
    print('=========================================================')
    print(f'total params: {number_format(total_params)}, '
          f'trainable params: {number_format(trainable_params)}, '
          f'flops: {number_format(flops_count)}')
    bat_x = paddle.rand(input_size)
    pred_y = model(bat_x)
    if isinstance(pred_y, tuple) or isinstance(pred_y, list):
        print(f'input shape: {bat_x.shape}, output shape: {[y.shape for y in pred_y]}')
    else:
        print(f'input shape: {bat_x.shape}, output shape: {pred_y.shape}')
