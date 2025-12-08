export class MinHeap<T> {
  private a: { k: number; v: T }[] = [];
  size() {
    return this.a.length;
  }
  push(k: number, v: T) {
    this.a.push({ k, v });
    this.bubbleUp(this.a.length - 1);
  }
  pop(): T | undefined {
    if (this.a.length === 0) return undefined;
    const top = this.a[0].v;
    const last = this.a.pop()!;
    if (this.a.length) {
      this.a[0] = last;
      this.bubbleDown(0);
    }
    return top;
  }
  peekKey(): number | undefined {
    return this.a[0]?.k;
  }
  private bubbleUp(i: number) {
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.a[p].k <= this.a[i].k) break;
      [this.a[p], this.a[i]] = [this.a[i], this.a[p]];
      i = p;
    }
  }
  private bubbleDown(i: number) {
    const n = this.a.length;
    while (true) {
      let l = i * 2 + 1,
        r = l + 1,
        m = i;
      if (l < n && this.a[l].k < this.a[m].k) m = l;
      if (r < n && this.a[r].k < this.a[m].k) m = r;
      if (m === i) break;
      [this.a[m], this.a[i]] = [this.a[i], this.a[m]];
      i = m;
    }
  }
}
