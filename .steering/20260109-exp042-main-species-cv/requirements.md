# Requirements

## 目的

exp040のCV戦略を改善し、全Main Speciesが全Foldに存在する分割を実現する

## 変更/追加する機能

- Main Species (species名の`_`分割1つ目) を用いたCV戦略
- グループ: sampling_date + state + main_species
- 層化: state + main_species
- seed: 8635

## 制約・前提条件

- exp040と同一モデル (DINOv3 Large Frozen)
- 3-fold維持
- 同一date+state+speciesは同じFoldに入る (データリーク防止)
