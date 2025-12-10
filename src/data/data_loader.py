import random

import numpy as np
import torch
import torch.nn.functional as functional
import yfinance as yf

stock_features = {
    "Open": 0,
    "High": 1,
    "Low": 2,
    "Close": 3,
    "Volume": 4,
    "Dividends": 5,
    # Extended features will be added dynamically during preprocessing
    "Midpoint": 6,  # (High + Low) / 2
    "MA_Open": 7,  # Moving Average of Open
    "MA_High": 8,  # Moving Average of High
    "MA_Low": 9,  # Moving Average of Low
    "MA_Close": 10,  # Moving Average of Close
    "MA_Volume": 11,  # Moving Average of Volume
    "MaxPool_High": 12,  # Max-pooled High over window
    "MaxPool_Low": 13,  # Max-pooled Low over window
    "MinPool_High": 14,  # Min-pooled High over window
    "MinPool_Low": 15,  # Min-pooled Low over window
    "Range": 16,  # High - Low (daily range)
    "MA_Range": 17,  # Moving Average of Range
}

stock_features_dict = {
    0: "Open",
    1: "High",
    2: "Low",
    3: "Close",
    4: "Volume",
    5: "Dividends",
    6: "Midpoint",
    7: "MA_Open",
    8: "MA_High",
    9: "MA_Low",
    10: "MA_Close",
    11: "MA_Volume",
    12: "MaxPool_High",
    13: "MaxPool_Low",
    14: "MinPool_High",
    15: "MinPool_Low",
    16: "Range",
    17: "MA_Range",
}

stock_names = {
    "apple": "AAPL",
    "google": "GOOGL",
    "meta": "META",
    "amazon": "AMZN",
    "tesla": "TSLA",
    "netflix": "NFLX",
    "microsoft": "MSFT",
    "ibm": "IBM",
    "intel": "INTC",
    "qualcomm": "QCOM",
    "amd": "AMD",
    "broadcom": "AVGO",
    "honeywell": "HON",
    "palantir": "PLTR",
    "tsmc": "TSM",
    "phizer": "PFE",
}


def preprocess_stock_data(data, ma_window=5, pool_window=3, add_features=None):
    """
    Preprocess stock data by adding technical indicators and engineered features.

    Parameters
    ----------
    data : torch.Tensor
        Input stock data of shape (t, c) where c is number of base features
        Expected order: [Open, High, Low, Close, Volume, Dividends]
    ma_window : int, default=5
        Window size for moving averages
    pool_window : int, default=3
        Window size for max/min pooling operations
    add_features : list, optional
        List of feature names to add. If None, adds all available features.
        Options: ['Midpoint', 'MA_Open', 'MA_High', 'MA_Low', 'MA_Close', 'MA_Volume',
                 'MaxPool_High', 'MaxPool_Low', 'MinPool_High', 'MinPool_Low', 'Range', 'MA_Range']

    Returns
    -------
    torch.Tensor
        Enhanced data with additional features of shape (t, C_extended)
    list
        List of feature names in order
    """
    if add_features is None:
        add_features = [
            "Midpoint",
            "MA_Open",
            "MA_High",
            "MA_Low",
            "MA_Close",
            "MA_Volume",
            "MaxPool_High",
            "MaxPool_Low",
            "MinPool_High",
            "MinPool_Low",
            "Range",
            "MA_Range",
        ]

    t, c = data.shape
    enhanced_data = data.clone()  # Start with original data
    feature_names = ["Open", "High", "Low", "Close", "Volume", "Dividends"]

    # Extract base features for calculations
    open_prices = data[:, 0]  # Open
    high_prices = data[:, 1]  # High
    low_prices = data[:, 2]  # Low
    close_prices = data[:, 3]  # Close
    volume = data[:, 4]  # Volume
    dividends = data[:, 5]  # Dividends  # noqa: F841

    # 1. Midpoint (High + Low) / 2
    if "Midpoint" in add_features:
        midpoint = (high_prices + low_prices) / 2.0
        enhanced_data = torch.cat([enhanced_data, midpoint.unsqueeze(1)], dim=1)
        feature_names.append("Midpoint")

    # 2. Moving Averages
    def compute_moving_average(series, window) -> torch.Tensor:
        """Compute moving average with proper padding for edge cases."""
        if len(series) < window:
            return series.clone()  # Return original if series too short

        # Use unfold to create sliding windows, then take mean
        padded = functional.pad(series.unsqueeze(0), (window - 1, 0), mode="replicate")
        windowed = padded.unfold(1, window, 1)
        return windowed.mean(dim=2).squeeze(0)

    if "MA_Open" in add_features:
        ma_open = compute_moving_average(open_prices, ma_window)
        enhanced_data = torch.cat([enhanced_data, ma_open.unsqueeze(1)], dim=1)
        feature_names.append("MA_Open")

    if "MA_High" in add_features:
        ma_high = compute_moving_average(high_prices, ma_window)
        enhanced_data = torch.cat([enhanced_data, ma_high.unsqueeze(1)], dim=1)
        feature_names.append("MA_High")

    if "MA_Low" in add_features:
        ma_low = compute_moving_average(low_prices, ma_window)
        enhanced_data = torch.cat([enhanced_data, ma_low.unsqueeze(1)], dim=1)
        feature_names.append("MA_Low")

    if "MA_Close" in add_features:
        ma_close = compute_moving_average(close_prices, ma_window)
        enhanced_data = torch.cat([enhanced_data, ma_close.unsqueeze(1)], dim=1)
        feature_names.append("MA_Close")

    if "MA_Volume" in add_features:
        ma_volume = compute_moving_average(volume, ma_window)
        enhanced_data = torch.cat([enhanced_data, ma_volume.unsqueeze(1)], dim=1)
        feature_names.append("MA_Volume")

    # 3. Max/Min Pooling operations
    def compute_pool_operation(series, window, operation="max") -> torch.Tensor:
        """Compute max or min pooling over sliding windows."""
        if len(series) < window:
            return series.clone()

        # Pad the series to handle edge cases
        padded = functional.pad(series.unsqueeze(0), (window - 1, 0), mode="replicate")
        windowed = padded.unfold(1, window, 1)

        if operation == "max":
            return windowed.max(dim=2)[0].squeeze(0)
        elif operation == "min":
            return windowed.min(dim=2)[0].squeeze(0)

        raise ValueError(f"Invalid operation '{operation}'. Expected 'max' or 'min'.")

    if "MaxPool_High" in add_features:
        maxpool_high = compute_pool_operation(high_prices, pool_window, "max")
        enhanced_data = torch.cat([enhanced_data, maxpool_high.unsqueeze(1)], dim=1)
        feature_names.append("MaxPool_High")

    if "MaxPool_Low" in add_features:
        maxpool_low = compute_pool_operation(low_prices, pool_window, "max")
        enhanced_data = torch.cat([enhanced_data, maxpool_low.unsqueeze(1)], dim=1)
        feature_names.append("MaxPool_Low")

    if "MinPool_High" in add_features:
        minpool_high = compute_pool_operation(high_prices, pool_window, "min")
        enhanced_data = torch.cat([enhanced_data, minpool_high.unsqueeze(1)], dim=1)
        feature_names.append("MinPool_High")

    if "MinPool_Low" in add_features:
        minpool_low = compute_pool_operation(low_prices, pool_window, "min")
        enhanced_data = torch.cat([enhanced_data, minpool_low.unsqueeze(1)], dim=1)
        feature_names.append("MinPool_Low")

    # 4. Range and Moving Average of Range
    if "Range" in add_features:
        daily_range = high_prices - low_prices
        enhanced_data = torch.cat([enhanced_data, daily_range.unsqueeze(1)], dim=1)
        feature_names.append("Range")

        if "MA_Range" in add_features:
            ma_range = compute_moving_average(daily_range, ma_window)
            enhanced_data = torch.cat([enhanced_data, ma_range.unsqueeze(1)], dim=1)
            feature_names.append("MA_Range")
    elif "MA_Range" in add_features:
        # Compute range first if MA_Range is requested but Range is not
        daily_range = high_prices - low_prices
        ma_range = compute_moving_average(daily_range, ma_window)
        enhanced_data = torch.cat([enhanced_data, ma_range.unsqueeze(1)], dim=1)
        feature_names.append("MA_Range")

    print(f"✓ Preprocessing complete: {c} → {enhanced_data.shape[1]} features")
    print(f"  Added features: {feature_names[c:]}")

    return enhanced_data, feature_names


def apply_preprocessing_to_stock_list(data_list, ma_window=5, pool_window=3, add_features=None):
    """
    Apply preprocessing to a list of stock data tensors.

    Parameters
    ----------
    data_list : List[torch.Tensor]
        List of stock data tensors
    ma_window : int, default=5
        Window size for moving averages
    pool_window : int, default=3
        Window size for pooling operations
    add_features : list, optional
        List of features to add

    Returns
    -------
    List[torch.Tensor]
        List of preprocessed stock data
    list
        Feature names (same for all stocks)
    """
    preprocessed_data = []
    feature_names = None

    for _, data in enumerate(data_list):
        enhanced_data, names = preprocess_stock_data(
            data, ma_window=ma_window, pool_window=pool_window, add_features=add_features
        )
        preprocessed_data.append(enhanced_data)

        if feature_names is None:
            feature_names = names

    print(f"✓ Preprocessing applied to {len(data_list)} stocks")
    return preprocessed_data, feature_names


def randbatchgen(data, bs, t_lookback, t_target, target_marg):  # data should be array
    # t_lookback lookback interval
    # t_target target interval
    t_max = data.shape[0]  # the length of time interval
    var = data.shape[1]  # number of variable/channel
    sp = random.sample(
        range(t_max - (t_lookback + t_target)), k=bs
    )  ## randomly generate starting point
    batch_data = torch.zeros(bs, t_lookback + t_target, var)
    label_data = torch.zeros(bs)
    for i in range(bs):
        batch_data[i, :, :] = data[sp[i] : sp[i] + (t_lookback + t_target), :]
        dec_price = data[sp[i] + t_lookback - 1, 0]  # we make decision based on open price
        max_future = torch.max(
            data[sp[i] + t_lookback : sp[i] + t_lookback + t_target, 1]
        )  # the max value can be reached within target window (recall index 1 is for high)
        if (
            max_future > dec_price * target_marg
        ):  # the rule is defined as whether we expect a profit with margin buy_th in a short-term (t_target)
            label_data[i] = 1  # indicates buy
        else:
            label_data[i] = 0  # indicates do not buy
    return batch_data, label_data


def randbatchgen5(args, data_list, bs, t_lookback, t_target, target_marg):
    """
    Generate training data

    Parameters
    ----------
    args : argparse.Namespace
        arguments parsed by argparse.
    data_list : List[torch.Tensor]
        List of stock data tensors, each of shape (t, c), where t and c denote the time and channel (features), respectively.
    bs : int
        Batch size
    t_lookback : int
        Lookback window size
    t_target : int
        Target window size
    target_marg : float
        Target margin for profit
    """
    t_max = data_list[0].shape[0]  # The interval of the training data
    var = data_list[0].shape[1]  # number of variable/channel
    num_stocks = len(data_list)  # number of stocks
    sp = random.sample(
        range(t_max - (t_lookback + t_target)), k=bs
    )  # randomly generates starting points
    batch_data = torch.zeros(bs, t_lookback + t_target, var * num_stocks)
    label_data = torch.zeros(bs, num_stocks)
    in_channel = [
        stock_features[f] for f in args.input_features
    ]  # List of input channels to be used for the model
    for i in range(bs):
        prices: dict[int, list[torch.Tensor]] = {ch: [] for ch in in_channel}

        for data in data_list:
            batch_segment = data[sp[i] : sp[i] + (t_lookback + t_target), :]  # (t, var)
            # don't shadow `i`; iterate real channel indices
            for ch in in_channel:
                prices[ch].append(batch_segment[:, ch].unsqueeze(1))  # (t,1)

        # now create stacked-per-feature tensors WITHOUT mutating the type of `prices`
        per_feature_stacked = [
            torch.cat(prices[ch], dim=1) for ch in in_channel
        ]  # each (t, num_stocks)

        # final (t, var*num_stocks) in the same order as in_channel
        batch_data[i, :, :] = torch.cat(per_feature_stacked, dim=1)

        for j, data in enumerate(data_list):
            # Find Open and High indices in the loaded features
            try:
                open_idx = args.input_features.index("Open")
                high_idx = args.input_features.index("High")
            except ValueError as error:
                raise ValueError(
                    "Both 'Open' and 'High' must be in input_features for label generation."
                ) from error

            dec_price = data[
                sp[i] + t_lookback - 1, open_idx
            ]  # we make decision based on open price
            max_future = torch.max(
                data[sp[i] + t_lookback : sp[i] + t_lookback + t_target, high_idx]
            )  # the max value can be reached within target window

            if (
                max_future > dec_price * target_marg
            ):  # the rule is defined as whether we expect a profit with margin buy_th in a short-term (t_target)
                label_data[i, j] = 1  # indicates buy
            else:
                label_data[i, j] = 0  # indicates do not buy
    return batch_data, label_data


def randbatchgen_v2(args, data_list, bs, t_lookback, t_target, target_marg):
    """
    Generate training data with improved stacking approach.

    Creates initial shape (bs, t_lookback + t_target, var, num_stocks) then
    concatenates each stock with their respective channels.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments parsed by argparse
    data_list : List[torch.Tensor]
        List of stock data tensors, each of shape (t, c)
    bs : int
        Batch size
    t_lookback : int
        Lookback window size
    t_target : int
        Target window size
    target_marg : float
        Target margin for profit calculation

    Returns
    -------
    torch.Tensor
        Batch data of shape (bs, t_lookback + t_target, var * num_stocks)
    torch.Tensor
        Label data of shape (bs, num_stocks)
    """
    t_max = data_list[0].shape[0]  # Time length of the data
    var = data_list[0].shape[1]  # Number of features/channels per stock
    num_stocks = len(data_list)  # Number of stocks

    # Randomly generate starting points for time windows
    sp = random.sample(range(t_max - (t_lookback + t_target)), k=bs)

    # Step 1: Create initial tensor with shape (bs, t_lookback + t_target, var, num_stocks)
    batch_data_4d = torch.zeros(bs, t_lookback + t_target, var, num_stocks)
    label_data = torch.zeros(bs, num_stocks)

    # Fill the 4D tensor
    for i in range(bs):
        for j, stock_data in enumerate(data_list):
            # Extract time window for this stock
            time_window = stock_data[sp[i] : sp[i] + (t_lookback + t_target), :]
            batch_data_4d[i, :, :, j] = time_window

            # Generate label for this stock
            # Find Open and High indices in the loaded features
            try:
                open_idx = args.input_features.index("Open")
                high_idx = args.input_features.index("High")
            except ValueError as error:
                raise ValueError(
                    f"Both 'Open' and 'High' must be in input_features for label generation. Missing: {error}"
                ) from error

            dec_price = stock_data[sp[i] + t_lookback - 1, open_idx]  # Decision price (Open)
            max_future = torch.max(
                stock_data[sp[i] + t_lookback : sp[i] + t_lookback + t_target, high_idx]
            )  # Max High in target window

            if max_future > dec_price * target_marg:
                label_data[i, j] = 1  # Buy signal
            else:
                label_data[i, j] = 0  # No buy signal

    # Step 2: Concatenate each stock with their respective channels
    # Reshape to (bs, t_lookback + t_target, var * num_stocks)
    # Each feature is grouped across all stocks: [Open_stock1, Open_stock2, ..., High_stock1, High_stock2, ...]
    batch_data_final = torch.zeros(bs, t_lookback + t_target, var * num_stocks)

    for i in range(bs):
        feature_concatenated = []

        # For each feature/channel
        for feature_idx in range(var):
            # Collect this feature across all stocks
            feature_across_stocks = batch_data_4d[i, :, feature_idx, :]  # Shape: (t, num_stocks)
            feature_concatenated.append(feature_across_stocks)

        # Concatenate features: [Open_all_stocks, High_all_stocks, Low_all_stocks, ...]
        batch_data_final[i] = torch.cat(feature_concatenated, dim=1)

    return batch_data_final, label_data.flatten()


def randbatchgen_v3(args, data_list, bs, t_lookback, t_target, target_marg, selected_features=None):
    """
    Function to generate data in the form of batch for training.

    It first collects the bs number of time series data in the shape of (t_lookback + t_target, number of features, number of stocks).

    Parameters
    ----------
    args : argparse.Namespace
        Arguments parsed by argparse
    data_list : List[torch.Tensor]
        List of stock data tensors, each of shape (t, c)
    bs : int
        Batch size
    t_lookback : int
        Lookback window size
    t_target : int
        Target window size
    target_marg : float
        Target margin for profit
    selected_features : list, optional
        List of feature names to include. If None, uses args.input_features

    Returns
    -------
    torch.Tensor
        Batch data with selected features concatenated across stocks
    torch.Tensor
        Label data of shape (bs, num_stocks)
    """
    t_max = data_list[0].shape[0]  # The total length of the time horizon.
    num_stocks = len(data_list)

    # Determine which features to use
    if selected_features is None:
        selected_features = (
            args.input_features
            if hasattr(args, "input_features")
            else ["Open", "High", "Low", "Close"]
        )

    # Get feature indices
    if hasattr(args, "feature_names") and args.feature_names:
        # Use the feature names from preprocessing if available
        feature_to_idx = {name: idx for idx, name in enumerate(args.feature_names)}
        selected_indices = [
            feature_to_idx[feat] for feat in selected_features if feat in feature_to_idx
        ]
    else:
        # Use loaded input features mapping (fixes the index out of bounds issue)
        loaded_feature_mapping = {feature: idx for idx, feature in enumerate(args.input_features)}
        selected_indices = [
            loaded_feature_mapping[feat]
            for feat in selected_features
            if feat in loaded_feature_mapping
        ]

    num_selected_features = len(selected_indices)

    """ Safety check: Ensure we have selected features
    # Debug information
    print(f"🔍 randbatchgen_v3 Debug:")
    print(f"  Selected features: {selected_features}")
    print(f"  Selected indices: {selected_indices}")
    print(f"  Data shape per stock: {data_list[0].shape}")
    print(f"  Args input features: {args.input_features if hasattr(args, 'input_features') else 'Not set'}")
    print(f"  Final output channels: {num_selected_features * num_stocks}")

    # Safety check: Compare with expected loaded feature indices (not global stock_features)
    if hasattr(args, 'input_features'):
        # Use the same loaded feature mapping for comparison
        loaded_feature_mapping_check = {feature: idx for idx, feature in enumerate(args.input_features)}
        expected_indices = [loaded_feature_mapping_check[feat] for feat in selected_features if feat in loaded_feature_mapping_check]
        if selected_indices != expected_indices:
            print(f"⚠️  WARNING: Feature mismatch!")
            print(f"  Expected (from loaded features): {expected_indices}")
            print(f"  Got (selected_indices): {selected_indices}")
            print(f"  This could cause MSE scale differences!")
        else:
            print(f"✅ Feature mapping is correct: {selected_indices}")
    """

    # Randomly generate starting points for training data
    sp = random.sample(range(t_max - (t_lookback + t_target)), k=bs)

    # Create 4D tensor: (bs, t, selected_features, num_stocks)
    batch_data_4d = torch.zeros(bs, t_lookback + t_target, num_selected_features, num_stocks)
    label_data = torch.zeros(bs, num_stocks)

    # Fill the tensor
    for i in range(bs):
        for j, stock_data in enumerate(data_list):
            # Extract time window
            time_window = stock_data[sp[i] : sp[i] + (t_lookback + t_target), :]

            # Select only the desired features
            selected_time_window = time_window[:, selected_indices]
            batch_data_4d[i, :, :, j] = selected_time_window

            # Generate labels using the actual indices from loaded data
            # Find Open and High indices in the loaded features
            try:
                open_idx = args.input_features.index("Open")
                high_idx = args.input_features.index("High")
            except ValueError as e:
                raise ValueError(
                    f"Both 'Open' and 'High' must be in input_features for label generation. Missing: {e}"
                ) from e

            dec_price = stock_data[
                sp[i] + t_lookback - 1, open_idx
            ]  # Opening price at the deay of buying stock option.
            max_future = torch.max(
                stock_data[sp[i] + t_lookback : sp[i] + t_lookback + t_target, high_idx]
            )  # The highest price observed in the t_target window
            # following the day the stock is bought.

            if (
                max_future > dec_price * target_marg
            ):  # Label "1" represents a minimum profit with the target margin.
                label_data[i, j] = 1
            else:
                label_data[i, j] = 0

    # Concatenate features across stocks
    batch_data_final = torch.zeros(bs, t_lookback + t_target, num_selected_features * num_stocks)

    for i in range(bs):
        feature_concatenated = []

        for feature_idx in range(num_selected_features):
            # Get this feature across all stocks
            feature_across_stocks = batch_data_4d[i, :, feature_idx, :]  # (t, num_stocks)
            feature_concatenated.append(feature_across_stocks)

        # Concatenate: [Feature1_all_stocks, Feature2_all_stocks, ...]
        batch_data_final[i] = torch.cat(feature_concatenated, dim=1)

    # Debug: Check data statistics
    # print(f"📊 Final batch statistics:")
    # print(f"  Batch shape: {batch_data_final.shape}")
    # print(f"  Data range: [{batch_data_final.min():.3f}, {batch_data_final.max():.3f}]")
    # print(f"  Data mean: {batch_data_final.mean():.3f}, std: {batch_data_final.std():.3f}")
    # print(f"  Label distribution: {label_data.sum().item()}/{len(label_data)} positive labels")

    return batch_data_final, label_data.flatten()


def randbatchgen_v4(args, data_list, bs, t_lookback, t_target, target_marg):
    """
    Simplified batch generator that exactly matches randbatchgen5 behavior.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments parsed by argparse
    data_list : List[torch.Tensor]
        List of stock data tensors, each of shape (t, c)
    bs : int
        Batch size
    t_lookback : int
        Lookback window size
    t_target : int
        Target window size
    target_marg : float
        Target margin for profit calculation

    Returns
    -------
    torch.Tensor
        Batch data of shape (bs, t_lookback + t_target, var * num_stocks)
    torch.Tensor
        Label data flattened to single dimension
    """
    t_max = data_list[0].shape[0]
    var = data_list[0].shape[1]
    num_stocks = len(data_list)
    sp = random.sample(range(t_max - (t_lookback + t_target)), k=bs)

    # Use the EXACT same logic as randbatchgen5
    batch_data = torch.zeros(bs, t_lookback + t_target, var * num_stocks)
    label_data = torch.zeros(bs, num_stocks)

    # Use the exact same feature processing as randbatchgen5
    in_channel = [stock_features[f] for f in args.input_features]

    for i in range(bs):
        prices_lists: dict[int, list[torch.Tensor]] = {ch: [] for ch in in_channel}

        for data in data_list:
            seg = data[sp[i] : sp[i] + (t_lookback + t_target), :]
            for ch in in_channel:
                prices_lists[ch].append(seg[:, ch].unsqueeze(1))

        # build once, don't reassign Tensor into a list-typed slot
        per_feature = [torch.cat(prices_lists[ch], dim=1) for ch in in_channel]
        batch_data[i, :, :] = torch.cat(per_feature, dim=1)

        # Generate labels exactly like randbatchgen5
        for j, data in enumerate(data_list):
            dec_price = data[sp[i] + t_lookback - 1, 0]
            max_future = torch.max(data[sp[i] + t_lookback : sp[i] + t_lookback + t_target, 1])

            if max_future > dec_price * target_marg:
                label_data[i, j] = 1
            else:
                label_data[i, j] = 0

    print("📊 randbatchgen_v4 (should match randbatchgen5):")
    print(f"  Batch shape: {batch_data.shape}")
    print(f"  Data range: [{batch_data.min():.3f}, {batch_data.max():.3f}]")
    print(f"  Data mean: {batch_data.mean():.3f}, std: {batch_data.std():.3f}")

    return batch_data, label_data.flatten()


def stack_randbatchgen(data_list, bs, t_lookback, t_target, target_marg):
    """
    Generate random batches from stacked stock data.
    Stocks are stacked consecutively.

    Parameters
    ----------
    data_list : List[torch.Tensor]
        List of stock data tensors, each of shape (t, c) where t is time steps and c is channels
    bs : int
        Batch size
    t_lookback : int
        Lookback window size
    t_target : int
        Target window size
    target_marg : float
        Target margin for profit calculation
    """
    # Stack all stocks along the channel dimension
    stacked_data = torch.cat(data_list, dim=1)  # Shape: (t, c * num_stocks)

    t_max = stacked_data.shape[0]  # Total time steps
    var = stacked_data.shape[1]  # Total number of channels (c * num_stocks)
    num_stocks = len(data_list)  # Number of stocks

    # Randomly generate starting points
    sp = random.sample(range(t_max - (t_lookback + t_target)), k=bs)

    # Initialize batch tensors
    batch_data = torch.zeros(bs, t_lookback + t_target, var)
    label_data = torch.zeros(bs, num_stocks)  # One label per stock

    for i in range(bs):
        # Get the batch segment
        batch_data[i, :, :] = stacked_data[sp[i] : sp[i] + (t_lookback + t_target), :]

        # Generate labels for each stock
        for j, data in enumerate(data_list):
            dec_price = data[sp[i] + t_lookback - 1, 0]  # Opening price at decision point
            max_future = torch.max(
                data[sp[i] + t_lookback : sp[i] + t_lookback + t_target, 1]
            )  # Max high price in target window

            # Label is 1 if we expect profit above target margin
            if max_future > dec_price * target_marg:
                label_data[i, j] = 1
            else:
                label_data[i, j] = 0

    return batch_data, label_data


def multi_stack_data(args):
    """
    Load and optionally preprocess stock data for training and testing.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments containing stock configuration and preprocessing options

    Returns
    -------
    tuple
        (train_data_list, test_data_list) - Lists of processed stock data tensors
    """
    train_stocks = args.train_stocks
    test_stocks = args.test_stocks
    tr_start, tr_end = args.train_start_date, args.train_end_date
    ts_start, ts_end = args.test_start_date, args.test_end_date
    in_channel = [f.capitalize() for f in args.input_features]
    train_data_list = []
    test_data_list = []
    all_stocks = train_stocks + test_stocks
    channels = np.asarray(in_channel)  # select values (open close etc.)

    # Check if all stocks are available
    for stock in all_stocks:
        if stock not in stock_names.keys():
            raise NotImplementedError("Stock not found ==>" + stock)

    # Download stock data
    for stock, id in stock_names.items():
        ticker = yf.Ticker(id)

        if stock in test_stocks:
            historical_data = ticker.history(start=ts_start, end=ts_end)
            data_test = historical_data[channels].to_numpy()
            data_tensor = torch.from_numpy(data_test).float()
            test_data_list.append(data_tensor)

        if stock in train_stocks:
            historical_data = ticker.history(start=tr_start, end=tr_end)  # Fixed: use train dates
            data_train = historical_data[channels].to_numpy()
            data_tensor = torch.from_numpy(data_train).float()
            train_data_list.append(data_tensor)

    print("Number of training stocks: ", len(train_data_list))
    print("Number of test stocks: ", len(test_data_list))
    assert len(train_data_list) == len(
        test_data_list
    ), "number of training stocks and test stocks are not equal"

    # Apply preprocessing if requested
    if hasattr(args, "use_preprocessing") and args.use_preprocessing:
        print("\n🔧 Applying preprocessing to stock data...")

        # Get preprocessing parameters from args
        ma_window = getattr(args, "ma_window", 5)
        pool_window = getattr(args, "pool_window", 3)
        add_features = getattr(args, "add_features", None)

        # Preprocess training data
        if train_data_list:
            train_data_list, train_feature_names = apply_preprocessing_to_stock_list(
                train_data_list,
                ma_window=ma_window,
                pool_window=pool_window,
                add_features=add_features,
            )
            print(f"Training data features: {train_feature_names}")

        # Preprocess test data
        if test_data_list:
            test_data_list, test_feature_names = apply_preprocessing_to_stock_list(
                test_data_list,
                ma_window=ma_window,
                pool_window=pool_window,
                add_features=add_features,
            )
            print(f"Test data features: {test_feature_names}")

        # Update args with new feature information
        if hasattr(args, "feature_names"):
            args.feature_names = train_feature_names if train_data_list else test_feature_names

        print("✅ Preprocessing completed successfully!")

    return train_data_list, test_data_list


def add_preprocessing_args(parser):
    """
    Add preprocessing arguments to argument parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Argument parser to add preprocessing options to
    """
    preprocessing_group = parser.add_argument_group("Preprocessing Options")

    preprocessing_group.add_argument(
        "--use_preprocessing",
        action="store_true",
        default=False,
        help="Enable feature preprocessing",
    )

    preprocessing_group.add_argument(
        "--ma_window", type=int, default=5, help="Window size for moving averages (default: 5)"
    )

    preprocessing_group.add_argument(
        "--pool_window", type=int, default=3, help="Window size for max/min pooling (default: 3)"
    )

    preprocessing_group.add_argument(
        "--add_features",
        nargs="*",
        choices=[
            "Midpoint",
            "MA_Open",
            "MA_High",
            "MA_Low",
            "MA_Close",
            "MA_Volume",
            "MaxPool_High",
            "MaxPool_Low",
            "MinPool_High",
            "MinPool_Low",
            "Range",
            "MA_Range",
        ],
        help="List of additional features to compute",
    )

    return parser


def get_extended_feature_mapping():
    """
    Get the extended feature mapping after preprocessing.

    Returns
    -------
    dict
        Updated stock_features dictionary with all possible features
    """
    return stock_features.copy()


def demo_preprocessing():
    """
    Demonstration of the preprocessing functionality.
    """
    print("🔧 Stock Data Preprocessing Demo")
    print("=" * 50)

    # Create sample data (t=100, c=6)
    t, c = 100, 6
    sample_data = torch.randn(t, c)
    sample_data[:, 1] = sample_data[:, 0] + torch.abs(torch.randn(t)) * 0.1  # High > Open
    sample_data[:, 2] = sample_data[:, 0] - torch.abs(torch.randn(t)) * 0.1  # Low < Open
    sample_data[:, 3] = sample_data[:, 0] + torch.randn(t) * 0.05  # Close near Open
    sample_data[:, 4] = torch.abs(torch.randn(t)) * 1000  # Volume
    sample_data[:, 5] = torch.zeros(t)  # Dividends

    print(f"Original data shape: {sample_data.shape}")

    # Test preprocessing with different feature sets
    test_cases = [
        (["Midpoint"], "Basic midpoint"),
        (["MA_Open", "MA_High", "MA_Low"], "Moving averages"),
        (["MaxPool_High", "MinPool_Low"], "Pooling operations"),
        (["Range", "MA_Range"], "Range features"),
        (None, "All features"),  # None means all features
    ]

    for features, description in test_cases:
        print(f"\n📊 {description}:")
        enhanced_data, feature_names = preprocess_stock_data(
            sample_data, ma_window=5, pool_window=3, add_features=features
        )
        print(f"  Shape: {sample_data.shape} → {enhanced_data.shape}")
        print(f"  Features: {feature_names}")


def label_gen(args, data, t_lookback, t_target, target_marg):
    t_max = data.shape[0]
    label_data = torch.zeros(t_max - (t_lookback + t_target))
    if args.binary_label:
        for i in range(t_max - (t_lookback + t_target)):
            dec_price = data[i + t_lookback - 1, 0]  # we make decision based on open price
            max_future = torch.max(
                data[i + t_lookback : i + t_lookback + t_target, 1]
            )  # the max value can be reached within target window (recall index 1 is for high)
            if (
                max_future > dec_price * target_marg
            ):  # the rule is defined as whether we expect a profit with margin buy_th in a short-term (t_target)
                label_data[i] = 1  # indicates buy
            else:
                label_data[i] = 0  # indicates do not buy
    else:
        for i in range(t_max - (t_lookback + t_target)):
            dec_price = data[i + t_lookback - 1, 0]
            max_future = torch.max(data[i + t_lookback : i + t_lookback + t_target, 1])
    return label_data
