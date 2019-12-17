from IPython.display import HTML
from autograd import numpy as np


def hidecode():
    return HTML('''<script>
    code_show=true; 
    function code_toggle() {
     if (code_show){
     $('div.input').hide();
     } else {
     $('div.input').show();
     }
     code_show = !code_show
    } 
    $( document ).ready(code_toggle);
    </script>
    The raw code for this IPython notebook is by default hidden for easier reading.
    To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')


def get_initialization(net, loader, optimizer, criterion, train_params, init_type = 0, callback = 0):
    """better to change net into model object
    """

    def apply_weights_init(type):
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Linear') != -1:
                fin = m.in_features
                fout =m.out_features
                y = 1.0 / np.sqrt(fin)
                gain = np.sqrt(2.0 / float(fin + fout)) * np.sqrt(3)
                if type == 0:
                    m.weight.data.uniform_(-gain, gain)
                elif type == 1:
                    m.weight.data.normal_(0.0, 1 / np.sqrt(y))
                m.bias.data.fill_(0)

        return weights_init

    if init_type != 2:
        net.apply(apply_weights_init(init_type))
    for epoch in range(train_params['epochs']):
        running_loss = 0
        for k, data in enumerate(loader, 0):
            X, y = data

            optimizer.zero_grad()
            outputs = net(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss
        if callback != 0 and epoch % callback == 0:
            print('[epoch %d] loss: %.3f' %
                  (epoch + 1, running_loss / callback/len(loader)))
    return net.state_dict()
