import trm
import utils
import torch
from torch import nn, optim


def train(model, enc_x, dec_x, target,
          epoches=1000, init_lr=0.1,
          n_step=100, gamma=0.1,
          device='cpu', show_loss=True,
          save=True, path=None):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, n_step, gamma)
    losses = [ ]
    for i in range(epoches):
        optimizer.zero_grad()
        output = model(enc_x, dec_x)
        loss = criterion(output, target.contiguous().view(-1))
        if (i + 1) % 10 == 0:
            print(f"Epoch {i + 1}/{epoches}  Loss: {loss.item():.4f}")
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()
    if show_loss:
        utils.draw_loss(losses)
    if save:
        torch.save(model, path)


@torch.no_grad()
def test(model, enc_x, dec_x, target, vocab):
    output = model(enc_x, dec_x)
    pred = output.argmax(dim=-1, keepdim=True)

    pred_list, tgt_list = [ ], [ ]
    for line in pred.view_as(target).cpu():
        s = ""
        for i in range(len(line)):
            s += list(vocab.keys())[ line[ i ] ] + ' '
        pred_list.append(s)
    for line in target.cpu():
        s = ""
        for i in range(len(line)):
            s += list(vocab.keys())[ line[ i ] ] + ' '
        tgt_list.append(s)

    print(f"target: \n{tgt_list}")
    print(f"pred: \n{pred_list}")


if __name__ == '__main__':
    config = utils.load_config()
    vocab = utils.load_vocab()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    e_x = [
        'I like you <pad>',
        "I very like you",
    ]
    d_x = [
        '<sta> I love you',
        '<sta> I love you',
    ]
    t_x = [
        'I love you <end>',
        'I love you <end>',
    ]
    e_x = utils.sen2vec(e_x).to(device)
    d_x = utils.sen2vec(d_x).to(device)
    t_x = utils.sen2vec(t_x).to(device)

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
    train(model, e_x, d_x, t_x, device=device, path=config['save_path'])
    test(model, e_x, d_x, t_x, vocab)
