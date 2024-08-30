from datetime import timedelta

from loguru import logger


def train_test_split_timebased(
    rating_score_df,
    args,
    user_id_col="user_id",
    timestamp_col="timestamp",
    remove_unseen_users_in_test=True,
):
    max_date = rating_score_df[timestamp_col].max().date()
    val_date = max_date - timedelta(days=args.test_num_days)
    train_date = val_date - timedelta(days=args.val_num_days)

    val_date_str = val_date.strftime("%Y-%m-%d")
    train_date_str = train_date.strftime("%Y-%m-%d")

    test_df = rating_score_df.loc[lambda df: df[timestamp_col].ge(val_date_str)]
    val_df = rating_score_df.loc[
        lambda df: df[timestamp_col].ge(train_date_str)
        & df[timestamp_col].lt(val_date_str)
    ]
    train_df = rating_score_df.loc[lambda df: df[timestamp_col].lt(train_date_str)]

    if remove_unseen_users_in_test:
        logger.info(f"Removing the new users in val and test sets...")
        train_users = train_df[user_id_col].unique()
        val_users_original = val_df[user_id_col].nunique()
        test_users_original = test_df[user_id_col].nunique()
        val_df = val_df.loc[lambda df: df[user_id_col].isin(train_users)]
        test_df = test_df.loc[lambda df: df[user_id_col].isin(train_users)]
        logger.info(
            f"Removed {val_users_original - val_df[user_id_col].nunique()} users from val set"
        )
        logger.info(
            f"Removed {test_users_original - test_df[user_id_col].nunique()} users from test set"
        )

    return train_df, val_df, test_df
