import trm
import utils
import torch
from torch import nn, optim


def train(model, enc_x, dec_x, target,
          epoches=4, lr=1e-4,
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
        print(f"Epoch {i + 1}/{epoches}  Loss: {loss.item()}")
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    if show_loss:
        utils.draw_loss(losses)
    if save:
        torch.save(model, path)


@torch.no_grad()
def test(model, enc_x, dec_x, target, vocab):
    output = model(enc_x, dec_x)
    pred = output.argmax(dim=-1, keepdim=True)

    pred_list = utils.vec2sen(pred.view_as(target).cpu(), vocab)
    tgt_list = utils.vec2sen(target.cpu(), vocab)

    print(f"target: \n{tgt_list}")
    print(f"pred: \n{pred_list}")


if __name__ == '__main__':
    config = utils.load_config()
    vocab = utils.load_vocab()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    e_x = [
        'I like you <pad>',
        'I very like you',
    ]
    d_x = [
        '<sta> I love you',
        '<sta> I love you',
    ]
    t_x = [
        'I love you <end>',
        'I love you <end>',
    ]
    e_x = utils.sen2vec(e_x, vocab).to(device)
    d_x = utils.sen2vec(d_x, vocab).to(device)
    t_x = utils.sen2vec(t_x, vocab).to(device)

    model = trm.Transformer(config[ 'n' ],
                            config[ 'n_vocab' ],
                            config[ 'd_model' ],
                            config[ 'd_k' ],
                            config[ 'd_v' ],
                            config[ 'n_head' ],
                            config[ 'd_ff' ],
                            config[ 'pad_token' ],
                            config[ 'max_len' ],
                            config[ 'dropout' ],
                            device)

    train(model, e_x, d_x, t_x, device=device, path=config[ 'save_path' ])
    # test(model, e_x, d_x, t_x, vocab)
    # model = torch.load(config['save_path'])
    test_dec_input = model.greedy_decoder(e_x, vocab[ '<sta>' ])
    print(f"greedy decoder input sequences:\n{utils.vec2sen(test_dec_input, vocab)}")
    test(model, e_x, test_dec_input, t_x, vocab)
