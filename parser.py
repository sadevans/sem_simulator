import argparse

def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', nargs='+', help='Разрешение маски: LowResolution для 768x1024 или HighResolution для 4096x6144')
    parser.add_argument('--frame', nargs='+', help='Фигуры, из которых состоит рамка. Сначала угловые, затем массивы.')
    parser.add_argument('--center', nargs='+', help='Изображения в центре рамки  и их количество')
    parser.add_argument('--figures', nargs='+', help='Фигуры, расположенные в свободном пространстве')

    return parser