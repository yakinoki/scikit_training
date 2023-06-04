import numpy as np

# データを生成する例
data = np.random.rand(100, 2)  # 100個の2次元データポイントを生成


from kmapper import KeplerMapper

# Kepler Mapperオブジェクトの初期化
mapper = KeplerMapper()

# データをマッピングする
graph = mapper.map(data)


from kmapper.plotlyviz import plotlyviz

# 可視化用のプロットを生成する
plotly_html = plotlyviz(graph, title="Kepler Mapper", color_function=data[:, 0])

# プロットを表示する
import plotly.offline as offline

offline.plot(plotly_html, filename="kepler_mapper_plot.html")
