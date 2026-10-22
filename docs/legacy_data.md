# Earlier public data and GEE materials

The archive [Traning Data of UNISON.zip](../Traning%20Data%20of%20UNISON.zip) retains its original filename and bytes so existing links remain valid. It predates this organized model release.

## Known data limitations

The archive contains six CSVs. `UNISON_MC_L0_GA_lowLAT_Terra.csv` and `UNISON_MC_L16_Bare_Terra.csv` have identical contents. The Terra vegetation-labelled entry cannot be treated as an independent vegetation training dataset. No replacement file is fabricated here.

Compare the intended training configuration and complete schema with the selected data; filenames alone do not establish compatibility. The six released bundles do not request TRI.

The archive's absolute temperature fields use Celsius. Dataset selection and settings differ across development versions. The archive and concise fitting helper are not an exact reconstruction of all upstream processing decisions or the six fitted model states.

## Earlier GEE links

The original README listed:

- [Earlier GEE training example](https://code.earthengine.google.com/e52cbc31f6d9fcf2228fa68f692c0171)
- [Earlier GEE application example](https://code.earthengine.google.com/77fdd8a4108c4a38179cabcc8971634a)

These are historical references. Their scripts, asset permissions, endpoint availability and compatibility with the current bundles have not been verified as a working online service. Use the new offline client for the supplied bundles. Future supported online links will appear in the [online registry](../online/endpoint_registry.json).
