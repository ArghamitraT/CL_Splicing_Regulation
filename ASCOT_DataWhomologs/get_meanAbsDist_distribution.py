
from utils import extract_mad_values, sample_values, plot_histogram, plot_kde, summarize_distribution, load_pkl


def main():

    division = "test"
    file_mad = f"/gpfs/commons/home/atalukder/Contrastive_Learning/data/ASCOT/{division}_ExonExon_meanAbsDist.pkl"

    # === Example usage ===

    mad_df = load_pkl(file_mad)
    mad_values = extract_mad_values(mad_df)
    mad_sample = sample_values(mad_values, n=5e6)
    plot_histogram(mad_sample)
    plot_kde(mad_sample)
    stats = summarize_distribution(mad_sample)
    print(stats)




if __name__ == "__main__":
    main()
