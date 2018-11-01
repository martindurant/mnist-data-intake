mnist-data-intake
-----------------

MNIST dataset and plugin for Intake.

This package includes an Intake catalogue pointing to remote MNIST label and
image files on `yann.lecun`_, and a short plugin to
read the untypical but almost-trivial data format in which it is stored,
`original code`_.

.. _yann.lecun: http://yann.lecun.com/exdb/mnist/
.. _original code: https://github.com/datapythonista/mnist/

After installation, the data will be available in the builtin catalogue,
``inake.cat.mnist``.