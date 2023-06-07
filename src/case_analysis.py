import polars as pl

def show_single_case(session_id: int, session_pl: pl.DataFrame, product_pl: pl.DataFrame, item_list_col='prev_items'):
    session_details = (
        session_pl.filter(pl.col('session_id')==session_id)
            .explode(item_list_col)
            .join(product_pl, left_on=[item_list_col, 'locale'], right_on=['id', 'locale'])
    ).collect()

    long_cols = ['title', 'desc']
    short_cols = ['unique_id', 'price', 'brand',
           'color', 'size', 'model', 'material', 'author', ]

    for idx, row in session_details.to_pandas().iterrows():
        print('+'*20)
        print(f"{idx}: {row[item_list_col]}")
        print('+'*20)
        for col in long_cols:
            print(f"==={col}====")
            print(' '*4 + f"{row[col]}")
            print()
        # print(row)
        print(row[short_cols])
        print()
        print()