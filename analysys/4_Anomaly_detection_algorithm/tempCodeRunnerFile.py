    plt.plot(df_score["play_idx"], df_score["score_norm"], label="Excitement Score")
    plt.scatter(df_score.loc[df_score["highlight"], "play_idx"],
                df_score.loc[df_score["highlight"], "score_norm"],
                color="red", label="Predicted Excitement")
    plt.xlabel("Play Index")
    plt.ylabel("Excitement Score (Normalized)")
    plt.title("Excitement Prediction Using IsolationForest")
    plt.legend()
    plt.show()