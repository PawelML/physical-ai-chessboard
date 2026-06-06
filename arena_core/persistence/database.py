from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from arena_core.persistence.models import Base


def create_engine(database_url: str) -> AsyncEngine:
    return create_async_engine(database_url, future=True)


def create_session_factory(database_url: str) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(create_engine(database_url), expire_on_commit=False)


async def init_db(database_url: str) -> None:
    engine = create_engine(database_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()


async def session_scope(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    async with session_factory() as session:
        async with session.begin():
            yield session
