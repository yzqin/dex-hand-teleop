# run this with
# sed -i -f replace_effect.sed *.dae
/<library_effects/,/<\/library_effects/d
/<library_images\/>/ {
  r materials.daelib
}