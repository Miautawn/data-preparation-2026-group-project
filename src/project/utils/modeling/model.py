import torch
import torch.nn as nn


class FitRecModel(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_sports: int,
        n_genders: int,
        n_sequential_features: int,
        use_heartrate_input: bool = True,
        seq_length: int = 10,
        embed_dim: int = 5,
        hidden_dim: int = 64,
        dropout_prob: int = 0.1,
    ):
        """Short-term heart rate prediction model.

        Args:
            n_users (int): how many unique users the model is personalizing for
            n_sports (int): how many unique sports are being considered
            n_genders (int): how many unique genders are being considered
            n_sequential_features (int): how many sequential features are being used
            use_heartrate_input (bool, optional): whether to use heart rate
                as an input feature for autoregressive modeling. Defaults to True.
            seq_length (int, optional): the length of the input sequences. Defaults to 10.
            embed_dim (int, optional): the dimension of the embedding space. Defaults to 5.
            hidden_dim (int, optional): the dimension of the hidden layers. Defaults to 64.
            dropout_prob (int, optional): the dropout probability. Defaults to 0.1.
        """
        super(FitRecModel, self).__init__()

        self.seq_length = seq_length

        # 1. Attribute Embeddings
        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.sport_embedding = nn.Embedding(n_sports, embed_dim)
        self.gender_embedding = nn.Embedding(n_genders, embed_dim)

        # 2. LSTM Sequential Processor
        self.input_dim = n_sequential_features + (embed_dim * 3)
        if use_heartrate_input:
            self.input_dim += 1  # Add one more feature for heart rate input

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0,  # No dropout between LSTM layers since we have only 1 layer
        )

        # 3. Dropout Layer
        self.dropout = nn.Dropout(dropout_prob)

        # 4. Final Prediction Layer (Regressor)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        seq_data: torch.tensor,
        user_id: torch.tensor,
        sport_id: torch.tensor,
        gender_id: torch.tensor,
    ):
        """Forward pass for the model.

        Args:
            seq_data (torch.tensor): tensor containing sequential inputs
                of shape: [batch_size, seq_length, n_sequential_features]

                Note: n_sequential_features should include heart rate input feature
                if using autoregressive modeling.

            user_id (torch.tensor): tensor containing user IDs
                shape: [batch_size, user_id]
            sport_id (torch.tensor): tensor containing sport IDs
                shape: [batch_size, sport_id]
            gender_id (torch.tensor): tensor containing gender IDs
                shape: [batch_size, gender_id]

        Returns:
            torch.tensor: tensor containing the model's predictions
                shape: [batch_size, 1]
        """

        # Generate and repeat static embeddings across all 10 time steps
        u_emb = self.user_embedding(user_id).unsqueeze(1).repeat(1, self.seq_length, 1)
        s_emb = (
            self.sport_embedding(sport_id).unsqueeze(1).repeat(1, self.seq_length, 1)
        )
        g_emb = (
            self.gender_embedding(gender_id).unsqueeze(1).repeat(1, self.seq_length, 1)
        )

        # Concatenate sequential and static features into single vector
        x = torch.cat([seq_data, u_emb, s_emb, g_emb], dim=2)

        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the LAST hidden state for short-term prediction
        last_hidden = self.dropout(h_n.squeeze(0))

        prediction = self.regressor(last_hidden)

        return prediction
