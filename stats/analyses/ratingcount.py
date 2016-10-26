from ..plotting import plot_histogram, plot_actual_histogram
import os

def generate(ratings):
    return ratings.groupby('user_id')['item_id'].count()

def plot(res, output_dir, ext='pdf'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    rpu_path = os.path.join(output_dir, "rpu.{0}".format(ext))
    plot_histogram(rpu_path, "Ratings por usuário", "ID do usuário",
                   res.keys(), "Número de ratings", res.values)
    cnt_path = os.path.join(output_dir, "count.{0}".format(ext))
    plot_actual_histogram(cnt_path, "Nº de usuários com determinado número de"
                          " ratings", res.values, "Nº de ratings",
                          "Nº de usuários")
