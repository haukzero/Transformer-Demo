import trm
import utils
import torch
from torch import nn, optim


def train(model, enc_x, dec_x, target,
          epoches=1000, lr=5e-5,
          device='cpu', show_loss=True,
          save=True, path=None):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = [ ]
    for i in range(epoches):
        optimizer.zero_grad()
        output = model(enc_x, dec_x)
        loss = criterion(output, target.contiguous().view(-1))
        if (i + 1) % 10 == 0:
            print(f"Epoch {i + 1}/{epoches}  Loss: {loss.item()}")
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    if show_loss:
        utils.draw_loss(losses)
    if save:
        torch.save(model, path)


if __name__ == '__main__':
    config = utils.load_config()
    vocab = utils.load_vocab()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    e_x = [
        'I very like you',
        'I very very very like you',
    ]
    d_x = [
        '<sta> I love you',
        '<sta> I very love you',
    ]
    t_x = [
        'I love you <end>',
        'I very love you <end>',
    ]
    e_x = utils.sen2vec(e_x, vocab, config[ 'max_len' ]).to(device)
    d_x = utils.sen2vec(d_x, vocab, config[ 'max_len' ]).to(device)
    t_x = utils.sen2vec(t_x, vocab, config[ 'max_len' ]).to(device)

    model = trm.Transformer(config[ 'n' ],
                            config[ 'n_vocab' ],
                            config[ 'd_model' ],
                            config[ 'd_k' ],
                            config[ 'd_v' ],
                            config[ 'n_head' ],
                            config[ 'd_ff' ],
                            config[ 'n_expert' ],
                            config[ 'd_exp' ],
                            config[ 'top_k' ],
                            config[ 'pad_token' ],
                            config[ 'max_len' ],
                            config[ 'dropout' ],
                            device)

    train(model, e_x, d_x, t_x, device=device, path=config[ 'save_path' ])
    # model = torch.load(config['save_path'])

    pred = model.greedy_decoder(e_x,
                                vocab[ '<sta>' ],
                                vocab[ '<end>' ],
                                vocab[ '<pad>' ])
    print(utils.vec2sen(pred, vocab))
