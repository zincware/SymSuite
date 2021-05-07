Installation
============
Here you will find everything you need to know about installing SymDet. Due to the status of the code the only install
option at the moment is from source. When the package has some more functionality and is better tested it will be
released on PyPi and Conda. For now, follow the instructions below and feel free to ask if you have any questions.

.. code-block:: bash

    git clone https://github.com/SamTov/SymDet
    cd SymDet
    pip3 install . --user

If you would like to edit the code or add some new features, I would recommend using the editable flag with pip as

.. code-block:: bash

    git clone https://github.com/SamTov/SymDet
    cd SymDet
    pip3 install -e . --user

This will point your python nasalisation to the SymDet directory in its current location rather than copying it into a
python package directory.