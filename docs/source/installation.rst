Installation
============
Here you will find everything you need to know about installing SymDet.

PyPi
****

.. code-block:: bash

   pip3 install symdet

Source
******

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

Local Documentation
*******************
At some stage you may want to look at the documentation locally.
This can be done simply by following the instructions below after having
cloned the repository:

.. code-block:: bash

   cd SymDet/docs
   make html
   firefox/chrome/safari/open Symdet/docs/build/html/index.html

Note, only one browser command should be used and the full path is included
only for clarity. If you already inside the docs directory just enter the rest
from there.