# sequential (a linear chain of modules)
#    to support multi-input,
# we support both packing and unpacking style for intermediates
#
# torch's own sequential is packing style.
#
# unpacking style may be useful in some cases as well.
#
# concat
# index
# loop (which contains sub op list)
# singleton (a single module)
#
# everything should have a shared namespace for simplicity,
# i.e. for loop op, we should make sure it's intermediates has no
# name collision with outer intermediate.
# to achieve this, we should pass the same intermediate over and over

